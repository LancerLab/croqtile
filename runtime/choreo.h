#ifndef __CHOREO_H__
#define __CHOREO_H__

#if __cplusplus < 201703L
// #error "Choreo requires C++17 or later"
#endif

#include <algorithm>
#include <assert.h>
#include <cmath>   // For fp16
#include <cstdint> // For fixed-width integer types
#include <cstdlib>
#include <initializer_list> // for std::initializer_list
#include <iostream>         // report error
#include <map>
#include <memory>
#include <random>
#include <sstream>
#include <string>

#if __has_include("private_target0_defines.h")
  #include "private_target0_defines.h"
#endif

#ifdef __CHOREO_PRIVATE_TGT0__

  #define __CHOREO_TARGET_NATIVE_F16_SUPPORT__
  // #define __CHOREO_TARGET_NATIVE_BF16_SUPPORT__
  #define __co_device__ __device__
  #define __co_host__ __host__
  #define __co_any__ __device__ __host__

#elif defined(__CHOREO_TARGET_CUTE__)
  #include <cuda_runtime.h>
  #include <filesystem>
  #include <fstream>
  #include <iterator>
  #include <limits.h>
  #include <utility>
  #ifdef __USE_CUDA_TYPE__
    #include "cuda.h"
    #if CUDA_VERSION >= 11000
      #define __CHOREO_TARGET_NATIVE_F16_SUPPORT__
      #include "cuda_bf16.h"
    #endif

    #define __CHOREO_TARGET_NATIVE_BF16_SUPPORT__
    #include "cuda_fp16.h"

    #define __CHOREO_TARGET_NATIVE_TF32_SUPPORT__

    #if CUDA_VERSION >= 11080
      #define __CHOREO_TARGET_NATIVE_FP8_SUPPORT__
      #if CUDA_VERSION >= 12090
        #define __CHOREO_TARGET_NATIVE_FP8_E8M0_SUPPORT__
      #endif
      #include "cuda_fp8.h"
    #else
    /* FP8 native types are only available when compiling for SM90+ targets. */
    #endif

    #if CUDA_VERSION >= 12090
      #define __CHOREO_TARGET_NATIVE_FP4_SUPPORT__
      #define __CHOREO_TARGET_NATIVE_FP6_SUPPORT__
      #include "cuda_fp4.h"
      #include "cuda_fp6.h"
    #else
      // Fallback to CUTE FP4/FP6 types when CUDA native types are unavailable.
      #define __CHOREO_TARGET_NATIVE_FP4_SUPPORT__
      #define __CHOREO_TARGET_NATIVE_FP6_SUPPORT__
    #endif

    #define __CHOREO_TARGET_NATIVE_SUB_BYTE_INTEGRAL_SUPPORT__
  #else // __USE_CUTE_TYPE__
    #define __CHOREO_TARGET_NATIVE_TF32_SUPPORT__
    #define __CHOREO_TARGET_NATIVE_F16_SUPPORT__
    #define __CHOREO_TARGET_NATIVE_BF16_SUPPORT__
    #define __CHOREO_TARGET_NATIVE_FP8_SUPPORT__
    #define __CHOREO_TARGET_NATIVE_FP6_SUPPORT__
    #define __CHOREO_TARGET_NATIVE_FP4_SUPPORT__
    #define __CHOREO_TARGET_NATIVE_SUB_BYTE_INTEGRAL_SUPPORT__
  #endif

  #include "cute/tensor.hpp"
  #include <cuda/barrier>
  #include <mma.h>
  #include <unistd.h>

  #define __co_device__ __device__
  #define __co_host__ __host__
  #define __co_any__ __device__ __host__

#else

  #define __co_device__
  #define __co_host__
  #define __co_any__

#endif // PRIVATE_TGT0 and CUTE

// private target must not enable native FP8 support
#if defined(__CHOREO_PRIVATE_TGT0__)
  #ifdef __CHOREO_TARGET_NATIVE_FP8_SUPPORT__
    #undef __CHOREO_TARGET_NATIVE_FP8_SUPPORT__
  #endif
  #ifdef __CHOREO_TARGET_NATIVE_FP8_E8M0_SUPPORT__
    #undef __CHOREO_TARGET_NATIVE_FP8_E8M0_SUPPORT__
  #endif
#endif

#if __CHOREO_TGT0_ARCH__ == 400
  #define __CHOREO_BLOCK_SINGLE__                                              \
    threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0 &&                \
        subThreadIdx.x == 0 && subThreadIdx.y == 0 && subThreadIdx.z == 0
  #define __CHOREO_GROUP_SINGLE__(GSIZE)                                       \
    subThreadIdx.x == 0 && subThreadIdx.y == 0 && subThreadIdx.z == 0
#elif defined(__CHOREO_TARGET_CUTE__)
  #define __CHOREO_BLOCK_SINGLE__                                              \
    threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0
  #define __CHOREO_GROUP_SINGLE__ threadIdx.x % 32 == 0
  #define __CHOREO_GROUPX4_SINGLE__ threadIdx.x % 128 == 0
  #define __CHOREO_GROUP_ID__                                                  \
    (threadIdx.x + threadIdx.y * blockDim.x +                                  \
     threadIdx.z * blockDim.x * blockDim.y) /                                  \
        32
#else
  #define __CHOREO_BLOCK_SINGLE__                                              \
    threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0
  #define __CHOREO_GROUP_SINGLE__ "invalid to use sublocal predicate"
#endif

#define __cok__ namespace choreo

namespace choreo {

constexpr size_t __inf__ = (size_t)((1LL << 32) - 1);

inline void __co_any__ choreo_assert(bool p, const char* msg,
                                     const char* file = __FILE__,
                                     int line = __LINE__) {
  if (!p) {
#ifdef __CHOREO_PRIVATE_TGT0__
    std::cerr << file << ":" << line << ": choreo assertion abort: " << msg
              << std::endl;
    std::abort();
#else
    printf("%s:%d: choreo assertion abort: %s\n", file, line, msg);
    assert(false);
#endif
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

#ifdef __CHOREO_PRIVATE_TGT0__
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
#endif // __CHOREO_PRIVATE_TGT0__

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

  SimpleArray(const SimpleArray&) = default;
  SimpleArray& operator=(const SimpleArray&) = default;
  ~SimpleArray() = default;

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

#ifdef __CHOREO_TARGET_NATIVE_TF32_SUPPORT__
  // TF32 is only used in tensor core in CUDA and CUTE
  #if defined(__USE_CUTE_TYPE__)
using cute::tfloat32_t;
  #elif defined(__USE_CUDA_TYPE__)
using tfloat32_t = nvcuda::wmma::precision::tf32;
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

#ifndef __CHOREO_TARGET_NATIVE_F16_SUPPORT__
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
  #elif defined(__CHOREO_PRIVATE_TGT0__)
using f16 = __fp16;
using half = __fp16;
  #else
    #error "half float is not supported on this target."
  #endif
#endif // __CHOREO_TARGET_NATIVE_F16_SUPPORT__

__co_any__ inline static f16 f32_to_f16(f32 value) {
#ifdef __USE_CUDA_TYPE__
  return __float2half(value);
#else
  return __f32_to_f16<f16>(value);
#endif
}

__co_any__ inline static f32 f16_to_f32(f16 value) {
#ifdef __USE_CUDA_TYPE__
  return __half2float(value);
#else
  return __f16_to_f32<f32>(value);
#endif
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

__co_any__ inline static bf16 f32_to_bf16(f32 value) {
  #ifdef __USE_CUDA_TYPE__
  return __float2bfloat16(value);
  #else
  return bf16(value);
  #endif
}

__co_any__ inline static f32 bf16_to_f32(bf16 value) {
  #ifdef __USE_CUDA_TYPE__
  return __bfloat162float(value);
  #else
  return f32(value);
  #endif
}

  // Check for __bf16 support
  #if !defined(__CHOREO_PRIVATE_TGT0__) && !defined(__clang__) &&              \
      !defined(__GNUC__) && !defined(__CUDACC__)
    #error                                                                     \
        "Compiler does not support __bf16. Please use a compiler that supports __bf16 or define a fallback type."
  #elif (defined(__clang__) && __clang_major__ < 11) ||                        \
      (defined(__GNUC__) && __GNUC__ < 11)
    #error                                                                     \
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
  #elif defined(__CHOREO_PRIVATE_TGT0__) || __CHOREO_TGT0_ARCH__ >= 400
  // TODO
  #else
    #error "FP8 E4M3 support requires CUTE Target."
  #endif
using f8 = float_e4m3_t; // define f8 as float_e4m3_t
using f8_e4m3 = float_e4m3_t;
using f8_e5m2 = float_e5m2_t;
using f8_ue4m3 = float_ue4m3_t;
using f8_ue8m0 = float_ue8m0_t;

  // Minimal arithmetic support for FP8 scalar types.
  // Choreo's codegen may form expressions like `fp8 + fp8` before casting.
  // CUTLASS/CUTE FP8 types and CUDA FP8 types don't consistently provide these
  // operators, so we define them here and return FP32.
  #if defined(__USE_CUDA_TYPE__)
__host__ __device__ static inline float operator+(__nv_fp8_e4m3 a,
                                                  __nv_fp8_e4m3 b) {
  return float(a) + float(b);
}
__host__ __device__ static inline float operator-(__nv_fp8_e4m3 a,
                                                  __nv_fp8_e4m3 b) {
  return float(a) - float(b);
}
__host__ __device__ static inline float operator*(__nv_fp8_e4m3 a,
                                                  __nv_fp8_e4m3 b) {
  return float(a) * float(b);
}
__host__ __device__ static inline float operator/(__nv_fp8_e4m3 a,
                                                  __nv_fp8_e4m3 b) {
  return float(a) / float(b);
}

__host__ __device__ static inline float operator+(__nv_fp8_e5m2 a,
                                                  __nv_fp8_e5m2 b) {
  return float(a) + float(b);
}
__host__ __device__ static inline float operator-(__nv_fp8_e5m2 a,
                                                  __nv_fp8_e5m2 b) {
  return float(a) - float(b);
}
__host__ __device__ static inline float operator*(__nv_fp8_e5m2 a,
                                                  __nv_fp8_e5m2 b) {
  return float(a) * float(b);
}
__host__ __device__ static inline float operator/(__nv_fp8_e5m2 a,
                                                  __nv_fp8_e5m2 b) {
  return float(a) / float(b);
}
  #endif // __USE_CUDA_TYPE__

#endif // __CHOREO_TARGET_NATIVE_FP8_SUPPORT__

#ifdef __CHOREO_TARGET_NATIVE_FP4_SUPPORT__
  #if defined(__USE_CUTE_TYPE__)
using cute::float_e2m1_t;
  #elif defined(__USE_CUDA_TYPE__)
    #if CUDA_VERSION >= 12090
using float_e2m1_t = __nv_fp4_e2m1;
    #else
using float_e2m1_t = cute::float_e2m1_t;
    #endif
  #elif defined(__CHOREO_PRIVATE_TGT0__) || __CHOREO_TGT0_ARCH__ >= 400
  // TODO
  #else
    #error "FP4 is not supported on this target."
  #endif
using f4_e2m1_t = float_e2m1_t;
using f4_e2m1 = float_e2m1_t;
#endif // __CHOREO_TARGET_NATIVE_FP4_SUPPORT__

#ifdef __CHOREO_TARGET_NATIVE_FP6_SUPPORT__
  #if defined(__USE_CUTE_TYPE__)
using cute::float_e2m3_t;
using cute::float_e3m2_t;
  #elif defined(__USE_CUDA_TYPE__)
    #if CUDA_VERSION >= 12090
using float_e3m2_t = __nv_fp6_e3m2;
using float_e2m3_t = __nv_fp6_e2m3;
    #else
using float_e3m2_t = cute::float_e3m2_t;
using float_e2m3_t = cute::float_e2m3_t;
    #endif
  #elif defined(__CHOREO_PRIVATE_TGT0__) || __CHOREO_TGT0_ARCH__ >= 400
  // TODO
  #else
    #error "FP6 is not supported on this target."
  #endif
using f6_e3m2_t = float_e3m2_t;
using f6_e2m3_t = float_e2m3_t;
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

template <typename T>
__co_any__ inline float to_f32(T value) {
  if constexpr (std::is_same<T, f64>::value) {
    return static_cast<float>(value);
  } else if constexpr (std::is_same<T, f32>::value) {
    return value;
  } else if constexpr (std::is_same<T, f16>::value) {
#ifndef __CHOREO_TARGET_NATIVE_F16_SUPPORT__
    return __f16_to_f32<float>(value);
#else
  #ifdef __USE_CUDA_TYPE__
    return __half2float(value);
  #else
    return static_cast<float>(value);
  #endif
#endif
  } else if constexpr (std::is_same<T, bf16>::value) {
#ifndef __CHOREO_TARGET_NATIVE_BF16_SUPPORT__
    return bf16::halfBitsToFloat(value);
#else
  #ifdef __USE_CUDA_TYPE__
    return __bfloat162float(value);
  #else
    return static_cast<float>(value);
  #endif
#endif
  } else if constexpr (
#ifdef __CHOREO_TARGET_NATIVE_FP8_SUPPORT__
      std::is_same<T, f8_e4m3>::value || std::is_same<T, f8_e5m2>::value ||
#endif
#ifdef __CHOREO_TARGET_NATIVE_FP6_SUPPORT__
      std::is_same<T, f6_e3m2>::value || std::is_same<T, f6_e2m3>::value ||
#endif
#ifdef __CHOREO_TARGET_NATIVE_FP4_SUPPORT__
      std::is_same<T, f4_e2m1>::value ||
#endif
#ifdef __CHOREO_TARGET_NATIVE_TF32_SUPPORT__
      std::is_same<T, tf32>::value ||
#endif
      std::is_integral<T>::value) {
    return static_cast<float>(value);
#ifdef __CHOREO_TARGET_NATIVE_SUB_BYTE_INTEGRAL_SUPPORT__
  } else if constexpr (std::is_same<T, uint4b_t>::value ||
                       std::is_same<T, uint6b_t>::value ||
                       std::is_same<T, uint2b_t>::value ||
                       std::is_same<T, uint1b_t>::value ||
                       std::is_same<T, int6b_t>::value ||
                       std::is_same<T, int4b_t>::value ||
                       std::is_same<T, int2b_t>::value ||
                       std::is_same<T, bin1_t>::value) {
    return static_cast<float>(static_cast<int>(value));
#endif
  } else {
    // todo: support more types
    static_assert(sizeof(T) == 0, "Unsupported type for to_f32 conversion.");
  }
}

template <typename A, typename B, typename C>
__co_host__ inline void
verify_matmul_row_col_subset(A& lhs, B& rhs, C& res, float base_tol,
                             float rel_tol, size_t max_i = 8,
                             size_t max_j = 8) {
  size_t m = res.shape()[0];
  size_t n = res.shape()[1];
  size_t k = lhs.shape()[1];
  size_t step_i = std::max<size_t>(1, m / max_i);
  size_t step_j = std::max<size_t>(1, n / max_j);
  for (size_t i = 0; i < m; i += step_i)
    for (size_t j = 0; j < n; j += step_j) {
      float ref = 0.0f;
      for (size_t kk = 0; kk < k; ++kk)
        ref += to_f32(lhs[(int)i][(int)kk]) * to_f32(rhs[(int)kk][(int)j]);
      float got = to_f32(res[(int)i][(int)j]);
      float tol = base_tol + rel_tol * std::abs(ref);
      choreo_assert(std::abs(got - ref) <= tol, "values are not equal.");
    }
}

template <typename A, typename B, typename C>
__co_host__ inline void
verify_matmul_row_row_subset(A& lhs, B& rhs, C& res, float base_tol,
                             float rel_tol, size_t max_i = 8,
                             size_t max_j = 8) {
  size_t m = res.shape()[0];
  size_t n = res.shape()[1];
  size_t k = lhs.shape()[1];
  size_t step_i = std::max<size_t>(1, m / max_i);
  size_t step_j = std::max<size_t>(1, n / max_j);
  for (size_t i = 0; i < m; i += step_i)
    for (size_t j = 0; j < n; j += step_j) {
      float ref = 0.0f;
      for (size_t kk = 0; kk < k; ++kk)
        ref += to_f32(lhs[(int)i][(int)kk]) * to_f32(rhs[(int)j][(int)kk]);
      float got = to_f32(res[(int)i][(int)j]);
      float tol = base_tol + rel_tol * std::abs(ref);
      choreo_assert(std::abs(got - ref) <= tol, "values are not equal.");
    }
}

// bitcast to uintx_t.
// The type in is_same is the underlying type not the type alias.
#if defined(__USE_CUDA_TYPE__)
template <typename T>
__host__ __device__ inline auto bitcast_uint(T x) {
  if constexpr (std::is_same<T, float>::value) return __float_as_uint(x);
  #if defined(__CHOREO_TARGET_NATIVE_TF32_SUPPORT__)
  else if constexpr (std::is_same<T, tf32>::value)
    return __float_as_uint(x);
  #endif // __CHOREO_TARGET_NATIVE_TF32_SUPPORT__
  #if defined(__CHOREO_TARGET_NATIVE_F16_SUPPORT__)
  else if constexpr (std::is_same<T, __half>::value)
    return __half_as_ushort(x);
  #endif // __CHOREO_TARGET_NATIVE_F16_SUPPORT__
  #if defined(__CHOREO_TARGET_NATIVE_BF16_SUPPORT__)
  else if constexpr (std::is_same<T, __nv_bfloat16>::value)
    return __bfloat16_as_ushort(x);
  #endif // __CHOREO_TARGET_NATIVE_BF16_SUPPORT__
  #if defined(__CHOREO_TARGET_NATIVE_FP8_SUPPORT__)
  else if constexpr (std::is_same<T, __nv_fp8_e4m3>::value ||
                     std::is_same<T, __nv_fp8_e5m2>::value)
    return static_cast<uint8_t>(x.__x);
  #endif // __CHOREO_TARGET_NATIVE_FP8_SUPPORT__
  #ifdef __CHOREO_TARGET_NATIVE_FP8_E8M0_SUPPORT__
  else if constexpr (std::is_same<T, __nv_fp8_e8m0>::value)
    return static_cast<uint8_t>(x.__x);
  #endif // __CHOREO_TARGET_NATIVE_FP8_E8M0_SUPPORT__
  else if constexpr (std::is_integral_v<T>) {
    static_assert(sizeof(T) == 1 || sizeof(T) == 2 || sizeof(T) == 4,
                  "integral T must be 1, 2, or 4 bytes");
    if constexpr (sizeof(T) == 1)
      return reinterpret_cast<uint8_t&>(x);
    else if constexpr (sizeof(T) == 2)
      return reinterpret_cast<uint16_t&>(x);
    else
      return reinterpret_cast<uint32_t&>(x);
  } else
    static_assert(sizeof(T) == 0, "Unsupported type for bitcast_uint");
}
#endif // defined(__USE_CUDA_TYPE__)
#if defined(__USE_CUTE_TYPE__)
template <typename T>
__host__ __device__ inline auto bitcast_uint(T x) {
  #if !defined(__CHOREO_TARGET_NATIVE_TF32_SUPPORT__) ||                       \
      !defined(__CHOREO_TARGET_NATIVE_F16_SUPPORT__) ||                        \
      !defined(__CHOREO_TARGET_NATIVE_BF16_SUPPORT__) ||                       \
      !defined(__CHOREO_TARGET_NATIVE_FP8_SUPPORT__)
    #error "All of the following macros must be defined: \
__CHOREO_TARGET_NATIVE_TF32_SUPPORT__, \
__CHOREO_TARGET_NATIVE_F16_SUPPORT__, \
__CHOREO_TARGET_NATIVE_BF16_SUPPORT__, \
__CHOREO_TARGET_NATIVE_FP8_SUPPORT__"
  #endif
  if constexpr (std::is_same<T, cute::float_e4m3_t>::value ||
                std::is_same<T, cute::float_e5m2_t>::value ||
                std::is_same<T, cute::float_ue4m3_t>::value ||
                std::is_same<T, cute::float_ue8m0_t>::value)
    return x.raw();
  else if constexpr (std::is_same<T, cute::half_t>::value ||
                     std::is_same<T, cute::bfloat16_t>::value)
    return x.raw();
  else if constexpr (std::is_same<T, float>::value ||
                     std::is_same<T, tf32>::value)
    return __float_as_uint(x);
  else if constexpr (std::is_integral_v<T>) {
    static_assert(sizeof(T) == 1 || sizeof(T) == 2 || sizeof(T) == 4,
                  "integral T must be 1, 2, or 4 bytes");
    if constexpr (sizeof(T) == 1)
      return reinterpret_cast<uint8_t&>(x);
    else if constexpr (sizeof(T) == 2)
      return reinterpret_cast<uint16_t&>(x);
    else
      return reinterpret_cast<uint32_t&>(x);
  } else
    static_assert(sizeof(T) == 0, "Unsupported type for bitcast_uint");
}
#endif // defined(__USE_CUTE_TYPE__)

#if defined(__CHOREO_TARGET_CUTE__)
// note: As long as result is of uint32_t type, then always using bitcast_u32
// will not incur any additional performance overhead.
template <typename T>
__host__ __device__ inline uint32_t bitcast_u32(T x) {
  return uint32_t(bitcast_uint(x));
}

template <typename T>
__host__ __device__ constexpr inline uint32_t broadcast_to_u32(T x) {
  static_assert(sizeof(T) == 1 || sizeof(T) == 2 || sizeof(T) == 4,
                "T must be 1, 2, or 4 bytes");

  if constexpr (sizeof(T) == 1) {
    return bitcast_u32(x) * 0x01010101U;
  } else if constexpr (sizeof(T) == 2) {
    uint32_t v = bitcast_u32(x);
    return (v << 16) | v;
  } else {
    return bitcast_u32(x);
  }
}
#endif // defined(__CHOREO_TARGET_CUTE__)

namespace utils {
template <typename U>
__co_host__ inline void fill_random(U* array, size_t N, U lb, U ub) {
  std::random_device rd;
  std::mt19937 gen(rd());

  if constexpr (std::is_integral<U>::value) {
    std::uniform_int_distribution<U> dist(lb, ub);
    std::generate_n(array, N, [&]() { return dist(gen); });
  } else if constexpr (std::is_floating_point<U>::value) {
    std::uniform_real_distribution<U> dist(lb, ub);
    std::generate_n(array, N, [&]() { return dist(gen); });
  } else if constexpr (std::is_same<U, f16>::value ||
                       std::is_same<U, bf16>::value ||
#ifdef __CHOREO_TARGET_NATIVE_TF32_SUPPORT__
                       std::is_same<U, tf32>::value ||
#endif
#ifdef __CHOREO_TARGET_NATIVE_FP8_SUPPORT__
                       std::is_same<U, f8_e4m3>::value ||
                       std::is_same<U, f8_e5m2>::value ||
#endif
#ifdef __CHOREO_TARGET_NATIVE_FP6_SUPPORT__
                       std::is_same<U, f6_e3m2>::value ||
                       std::is_same<U, f6_e2m3>::value ||
#endif
#ifdef __CHOREO_TARGET_NATIVE_FP4_SUPPORT__
                       std::is_same<U, f4_e2m1>::value ||
#endif
                       false) {
    std::uniform_real_distribution<float> dist(to_f32(lb), to_f32(ub));
    std::generate_n(array, N, [&]() { return U(dist(gen)); });
  } else if constexpr (
#ifdef __CHOREO_TARGET_NATIVE_SUB_BYTE_INTEGRAL_SUPPORT__
      std::is_same<U, uint4b_t>::value || std::is_same<U, uint6b_t>::value ||
      std::is_same<U, uint2b_t>::value || std::is_same<U, uint1b_t>::value ||
      std::is_same<U, int6b_t>::value || std::is_same<U, int4b_t>::value ||
      std::is_same<U, int2b_t>::value || std::is_same<U, bin1_t>::value ||
#endif
      false) {
    std::uniform_int_distribution<int> dist((int)lb, (int)ub);
    std::generate_n(array, N, [&]() { return U(dist(gen)); });
  } else {
    static_assert(sizeof(U) == 0, "Unsupported type for fill_random.");
  }
}

// Helper to check if a value is zero
template <typename U>
__co_host__ inline bool is_zero(const U& v) {
  if constexpr (std::is_integral<U>::value) {
    return v == 0;
  } else if constexpr (std::is_floating_point<U>::value) {
    return v == static_cast<U>(0);
  } else if constexpr (std::is_same<U, f16>::value ||
                       std::is_same<U, bf16>::value ||
#ifdef __CHOREO_TARGET_NATIVE_FP8_SUPPORT__
                       std::is_same<U, f8_e4m3>::value ||
                       std::is_same<U, f8_e5m2>::value ||
#endif
                       false) {
    return to_f32(v) == 0.0f;
  } else {
    return false;
  }
}

// Fill a 2:4 structured sparse matrix (rank-2) with the first 2 of every 4 =
// nonzero
template <typename U>
__co_host__ inline void fill_ss(U* array, size_t M, size_t K,
                                U nonzero = U(1)) {
  for (size_t i = 0; i < M; ++i) {
    for (size_t k4 = 0; k4 < K / 4; ++k4) {
      size_t base = i * K + k4 * 4;
      array[base + 0] = nonzero;
      array[base + 1] = nonzero;
      array[base + 2] = U(0);
      array[base + 3] = U(0);
    }
  }
}

// Fill a 2:4 structured sparse matrix (rank-2) with random values
template <typename U>
__co_host__ inline void fill_random_ss(U* array, size_t M, size_t K, U lb,
                                       U ub) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<int> pick_pos(0, 3);
  for (size_t i = 0; i < M; ++i) {
    for (size_t k4 = 0; k4 < K / 4; ++k4) {
      size_t base = i * K + k4 * 4;
      // pick two distinct positions
      int p0 = pick_pos(gen);
      int p1 = pick_pos(gen);
      while (p1 == p0) p1 = pick_pos(gen);
      if (p1 < p0) std::swap(p0, p1);
      for (int p = 0; p < 4; ++p) array[base + p] = U(0);
      U v0 = U(1);
      U v1 = U(1);
      if constexpr (std::is_integral<U>::value) {
        std::uniform_int_distribution<int> dist((int)lb, (int)ub);
        v0 = U(dist(gen));
        v1 = U(dist(gen));
      } else if constexpr (std::is_floating_point<U>::value) {
        std::uniform_real_distribution<float> dist((float)lb, (float)ub);
        v0 = U(dist(gen));
        v1 = U(dist(gen));
      } else {
        std::uniform_real_distribution<float> dist(to_f32(lb), to_f32(ub));
        v0 = U(dist(gen));
        v1 = U(dist(gen));
      }
      if (is_zero(v0)) v0 = U(1);
      if (is_zero(v1)) v1 = U(1);
      array[base + p0] = v0;
      array[base + p1] = v1;
    }
  }
}

// Encode 2:4 structured sparse data into packed values and metadata
// Input: dense [M, K] with 2:4 sparsity pattern
// Output: packed [M, K/2] values, metadata [M, K/4] byte masks
template <typename U>
__co_host__ inline void encode_sparse_2to4(const U* dense, U* packed,
                                           uint8_t* metadata, size_t M,
                                           size_t K) {
  const size_t chunks = K / 4;
  for (size_t i = 0; i < M; ++i) {
    for (size_t k4 = 0; k4 < chunks; ++k4) {
      size_t base = i * K + k4 * 4;
      size_t out_base = i * (K / 2) + k4 * 2;
      U a0 = dense[base + 0];
      U a1 = dense[base + 1];
      U a2 = dense[base + 2];
      U a3 = dense[base + 3];
      uint8_t nibble = 0;
      int count = 0;
      int idxs[2] = {0, 0};
      if (!is_zero(a0) && count < 2) {
        packed[out_base + count++] = a0;
        idxs[count - 1] = 0;
      }
      if (!is_zero(a1) && count < 2) {
        packed[out_base + count++] = a1;
        idxs[count - 1] = 1;
      }
      if (!is_zero(a2) && count < 2) {
        packed[out_base + count++] = a2;
        idxs[count - 1] = 2;
      }
      if (!is_zero(a3) && count < 2) {
        packed[out_base + count++] = a3;
        idxs[count - 1] = 3;
      }
      while (count < 2) packed[out_base + count++] = U(0);
      nibble = (idxs[0] & 0x3) | ((idxs[1] & 0x3) << 2);
      metadata[i * chunks + k4] = nibble;
    }
  }
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

  // FP8-friendly fill: accepts float and converts for fp8 types.
  __co_any__ void fill_fp8(float value) {
#ifdef __CHOREO_TARGET_NATIVE_FP8_SUPPORT__
    if constexpr (std::is_same<T, f8_e4m3>::value ||
                  std::is_same<T, f8_e5m2>::value) {
      fill_n(this->data(), this->element_count(), T(value));
      return;
    }
#endif
    fill_n(this->data(), this->element_count(), static_cast<T>(value));
  }

  template <typename U>
  __co_host__ void fill_random(U lb, U ub) {
    utils::fill_random(this->data(), this->element_count(), static_cast<T>(lb),
                       static_cast<T>(ub));
  }

  // FP8-friendly random fill with float bounds.
  __co_host__ void fill_random_fp8(float lb, float ub) {
#ifdef __CHOREO_TARGET_NATIVE_FP8_SUPPORT__
    if constexpr (std::is_same<T, f8_e4m3>::value ||
                  std::is_same<T, f8_e5m2>::value) {
      utils::fill_random(this->data(), this->element_count(), T(lb), T(ub));
      return;
    }
#endif
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

  // FP8-friendly fill: accepts float and converts for fp8 types.
  __co_any__ void fill_fp8(float value) {
#ifdef __CHOREO_TARGET_NATIVE_FP8_SUPPORT__
    if constexpr (std::is_same<T, f8_e4m3>::value ||
                  std::is_same<T, f8_e5m2>::value) {
      fill_n(this->data(), this->element_count(), T(value));
      return;
    }
#endif
    fill_n(this->data(), this->element_count(), static_cast<T>(value));
  }

  template <typename U>
  __co_host__ void fill_random(U lb, U ub) {
    utils::fill_random(this->data(), this->element_count(), static_cast<T>(lb),
                       static_cast<T>(ub));
  }

  // FP8-friendly random fill with float bounds.
  __co_host__ void fill_random_fp8(float lb, float ub) {
#ifdef __CHOREO_TARGET_NATIVE_FP8_SUPPORT__
    if constexpr (std::is_same<T, f8_e4m3>::value ||
                  std::is_same<T, f8_e5m2>::value) {
      utils::fill_random(this->data(), this->element_count(), T(lb), T(ub));
      return;
    }
#endif
    utils::fill_random(this->data(), this->element_count(), static_cast<T>(lb),
                       static_cast<T>(ub));
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

// remove const version
template <typename T, size_t Rank>
__co_any__ spanned_view<typename std::remove_const<T>::type, Rank>
make_spanview(T* ptr, std::initializer_list<size_t> init) {
  using U = typename std::remove_const<T>::type;
  return spanned_view<U, Rank>(const_cast<U*>(ptr), make_mdspan<Rank>(init));
}

// void* version
template <typename T, size_t Rank>
__co_any__ spanned_view<typename std::remove_const<T>::type, Rank>
make_spanview(void* ptr, std::initializer_list<size_t> init) {
  static_assert(!std::is_void<T>::value, "T must not be void");
  static_assert(std::is_object<T>::value, "T must be an object type");
  using U = typename std::remove_const<T>::type;
  return spanned_view<U, Rank>(reinterpret_cast<U*>(ptr),
                               make_mdspan<Rank>(init));
}

// const void* version
template <typename T, size_t Rank>
__co_any__ spanned_view<typename std::remove_const<T>::type, Rank>
make_spanview(const void* ptr, std::initializer_list<size_t> init) {
  static_assert(!std::is_void<T>::value, "T must not be void");
  static_assert(std::is_object<T>::value, "T must be an object type");
  using U = typename std::remove_const<T>::type;
  return spanned_view<U, Rank>(const_cast<U*>(reinterpret_cast<const U*>(ptr)),
                               make_mdspan<Rank>(init));
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
#ifdef __CHOREO_PRIVATE_TGT0__
  // host memory optimization
  runtime_check(!tgt0HostMalloc(&raw_ptr, element_count * sizeof(T)),
                "[choreo-rt] failed to allocate memory.");
  auto del = [](T* p) {
    runtime_check(!tgt0HostFree(p), "[choreo-rt] failed to free memory.");
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

namespace utils {
template <typename U>
__co_any__ inline U from_f32(float v) {
  if constexpr (std::is_same<U, f16>::value) {
    return f16(v);
  } else if constexpr (std::is_same<U, bf16>::value) {
    return bf16(v);
#ifdef __CHOREO_TARGET_NATIVE_FP8_SUPPORT__
  } else if constexpr (std::is_same<U, f8_e4m3>::value) {
  #ifdef __USE_CUDA_TYPE__
    return f8_e4m3(v);
  #else
    return f8_e4m3(v);
  #endif
  } else if constexpr (std::is_same<U, f8_e5m2>::value) {
  #ifdef __USE_CUDA_TYPE__
    return f8_e5m2(v);
  #else
    return f8_e5m2(v);
  #endif
#endif
  } else {
    return static_cast<U>(v);
  }
}

// Host-side 2:4 sparse init/encode trait with metadata packed by META_K groups.
template <typename ValueT, typename MetaT, size_t META_K>
struct Sparse2to4HostPolicy {
  static_assert(META_K % 4 == 0, "META_K must be a multiple of 4.");
  static constexpr size_t groups_per_strip = META_K / 4;
  static constexpr size_t bits_per_strip = groups_per_strip * 4;
  static constexpr bool meta_ok =
      (bits_per_strip <= sizeof(MetaT) * 8) ||
      (META_K == 64 && std::is_same<MetaT, choreo::u32>::value);
  static_assert(meta_ok, "MetaT is too small for META_K.");

  __co_host__ static inline void
  init_structured_sparse_A(spanned_data<ValueT, 2>& dense, std::mt19937& gen) {
    const size_t M = dense.shape()[0];
    const size_t K = dense.shape()[1];
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::uniform_int_distribution<int> pick(0, 3);
#ifdef __CHOREO_TARGET_NATIVE_FP8_SUPPORT__
    if constexpr (std::is_same<ValueT, f8_e4m3>::value ||
                  std::is_same<ValueT, f8_e5m2>::value) {
      std::memset(dense.data(), 0, dense.bytes());
    } else
#endif
    {
      dense.fill(ValueT(0));
    }
    for (size_t r = 0; r < M; ++r) {
      for (size_t c_group = 0; c_group < K / 4; ++c_group) {
        int idx0 = pick(gen);
        int idx1 = pick(gen);
        while (idx1 == idx0) idx1 = pick(gen);
        if (idx0 > idx1) std::swap(idx0, idx1);
        float v0 = dist(gen);
        float v1 = dist(gen);
        if (v0 == 0.0f) v0 = 1.0f;
        if (v1 == 0.0f) v1 = -1.0f;
        size_t base = r * K + c_group * 4;
        ValueT t0 = from_f32<ValueT>(v0);
        ValueT t1 = from_f32<ValueT>(v1);
#ifdef __CHOREO_TARGET_NATIVE_FP8_SUPPORT__
        if constexpr (std::is_same<ValueT, f8_e4m3>::value ||
                      std::is_same<ValueT, f8_e5m2>::value) {
          auto fp8_is_zero = [](const ValueT& v) {
            const uint8_t* p = reinterpret_cast<const uint8_t*>(&v);
            uint64_t raw = 0;
            for (size_t bi = 0; bi < sizeof(ValueT); ++bi)
              raw |= (uint64_t(p[bi]) << (8 * bi));
            const uint64_t neg_zero = (uint64_t(1) << (sizeof(ValueT) * 8 - 1));
            return (raw == 0) || (raw == neg_zero);
          };
          if (fp8_is_zero(t0)) t0 = from_f32<ValueT>(1.0f);
          if (fp8_is_zero(t1)) t1 = from_f32<ValueT>(-1.0f);
        } else
#endif
        {
          if (is_zero(t0)) t0 = from_f32<ValueT>(1.0f);
          if (is_zero(t1)) t1 = from_f32<ValueT>(-1.0f);
        }
        dense.data()[base + idx0] = t0;
        dense.data()[base + idx1] = t1;
      }
    }
  }

  // Initialize 2:4 structured sparse A with all nonzero values = 1.0.
  // Positions can be fixed (pos0,pos1) or random (random_pos=true).
  __co_host__ static inline void
  init_structured_sparse_A_ones(spanned_data<ValueT, 2>& dense,
                                std::mt19937& gen, bool random_pos = false,
                                int pos0 = 0, int pos1 = 1) {
    const size_t M = dense.shape()[0];
    const size_t K = dense.shape()[1];
    choreo_assert(pos0 >= 0 && pos0 < 4 && pos1 >= 0 && pos1 < 4,
                  "invalid sparse positions", __FILE__, __LINE__);
    choreo_assert(pos0 != pos1, "sparse positions must differ", __FILE__,
                  __LINE__);
    if (pos0 > pos1) std::swap(pos0, pos1);
    std::uniform_int_distribution<int> pick(0, 3);
#ifdef __CHOREO_TARGET_NATIVE_FP8_SUPPORT__
    if constexpr (std::is_same<ValueT, f8_e4m3>::value ||
                  std::is_same<ValueT, f8_e5m2>::value) {
      std::memset(dense.data(), 0, dense.bytes());
    } else
#endif
    {
      dense.fill(ValueT(0));
    }
    for (size_t r = 0; r < M; ++r) {
      for (size_t c_group = 0; c_group < K / 4; ++c_group) {
        int idx0 = pos0;
        int idx1 = pos1;
        if (random_pos) {
          idx0 = pick(gen);
          idx1 = pick(gen);
          while (idx1 == idx0) idx1 = pick(gen);
          if (idx0 > idx1) std::swap(idx0, idx1);
        }
        size_t base = r * K + c_group * 4;
        ValueT t = from_f32<ValueT>(1.0f);
#ifdef __CHOREO_TARGET_NATIVE_FP8_SUPPORT__
        if constexpr (std::is_same<ValueT, f8_e4m3>::value ||
                      std::is_same<ValueT, f8_e5m2>::value) {
          auto fp8_is_zero = [](const ValueT& v) {
            const uint8_t* p = reinterpret_cast<const uint8_t*>(&v);
            uint64_t raw = 0;
            for (size_t bi = 0; bi < sizeof(ValueT); ++bi)
              raw |= (uint64_t(p[bi]) << (8 * bi));
            const uint64_t neg_zero = (uint64_t(1) << (sizeof(ValueT) * 8 - 1));
            return (raw == 0) || (raw == neg_zero);
          };
          if (fp8_is_zero(t)) t = from_f32<ValueT>(1.0f);
        } else
#endif
        {
          if (is_zero(t)) t = from_f32<ValueT>(1.0f);
        }
        dense.data()[base + idx0] = t;
        dense.data()[base + idx1] = t;
      }
    }
  }

  // Encode 2:4 sparse A into packed values and META_K-grouped metadata.
  __co_host__ static inline void
  encode(spanned_data<ValueT, 2>& dense, spanned_data<ValueT, 2>& packed,
         spanned_data<MetaT, 2>& meta, std::vector<MetaT>* row_meta = nullptr) {
    const size_t M = dense.shape()[0];
    const size_t K = dense.shape()[1];
    const size_t strips = K / META_K;
    const bool fp8_k64_u32 =
        (META_K == 64 && std::is_same<MetaT, choreo::u32>::value);
    const size_t meta_cols = strips * (fp8_k64_u32 ? 2 : 1);
    if (row_meta) row_meta->assign(M * meta_cols, MetaT(0));
    for (size_t r = 0; r < M; ++r) {
      for (size_t strip = 0; strip < strips; ++strip) {
        uint64_t meta_val64 = 0;
        uint32_t meta_lo = 0;
        uint32_t meta_hi = 0;
        for (size_t cg = 0; cg < groups_per_strip; ++cg) {
          size_t c_group = strip * groups_per_strip + cg;
          size_t base = r * K + c_group * 4;
          size_t in_base = r * (K / 2) + c_group * 2;
          int idxs[2] = {-1, -1};
          int nz = 0;
          for (int i = 0; i < 4; ++i) {
            bool nonzero = false;
#ifdef __CHOREO_TARGET_NATIVE_FP8_SUPPORT__
            if constexpr (std::is_same<ValueT, f8_e4m3>::value ||
                          std::is_same<ValueT, f8_e5m2>::value) {
              const uint8_t* p =
                  reinterpret_cast<const uint8_t*>(&dense.data()[base + i]);
              uint64_t raw = 0;
              for (size_t bi = 0; bi < sizeof(ValueT); ++bi)
                raw |= (uint64_t(p[bi]) << (8 * bi));
              const uint64_t neg_zero =
                  (uint64_t(1) << (sizeof(ValueT) * 8 - 1));
              nonzero = (raw != 0) && (raw != neg_zero);
            } else
#endif
            {
              nonzero = (to_f32(dense.data()[base + i]) != 0.0f);
            }
            if (nonzero) {
              if (nz < 2) idxs[nz] = i;
              nz++;
            }
          }
          if constexpr (META_K == 64) {
#ifdef __CHOREO_TARGET_NATIVE_FP8_SUPPORT__
            if constexpr (std::is_same<ValueT, f8_e4m3>::value ||
                          std::is_same<ValueT, f8_e5m2>::value) {
              // Handle corner cases to match SM90 legacy compressor behavior
              if (nz == 1) {
                int only = idxs[0];
                if (only == 3) {
                  idxs[0] = 0;
                  idxs[1] = 3;
                } else {
                  idxs[0] = only;
                  idxs[1] = 3;
                }
                nz = 2;
              } else if (nz == 0) {
                idxs[0] = 0;
                idxs[1] = 3;
                nz = 2;
              } else if (nz != 2) {
                // keep first two if overfull
                nz = 2;
              }
            } else
#endif
            {
              if (nz != 2) {
                // Fallback to a deterministic pair to avoid abort during debug
                idxs[0] = 0;
                idxs[1] = 1;
                nz = 2;
              }
            }
          } else {
            choreo_assert(nz == 2, "Invalid 2:4 structure");
          }
          if (idxs[0] > idxs[1]) std::swap(idxs[0], idxs[1]);
          ValueT v0 = dense.data()[base + idxs[0]];
          ValueT v1 = dense.data()[base + idxs[1]];
#ifdef __CHOREO_TARGET_NATIVE_FP8_SUPPORT__
          if constexpr (std::is_same<ValueT, f8_e4m3>::value ||
                        std::is_same<ValueT, f8_e5m2>::value) {
            // If we synthesized idxs for nz<2, force zeros at the synthetic
            // slot
            if (nz == 2) {
              if (idxs[0] == 0 && idxs[1] == 3) {
                // preserve existing values
              }
            }
          }
#endif
          packed.data()[in_base + 0] = v0;
          packed.data()[in_base + 1] = v1;
          int pair_idx = static_cast<int>(cg) * 2;
          // ordered_metadata uses the same 2-bit index encoding; order is
          // enforced by sorted idxs.
          uint32_t nibble =
              (uint32_t(idxs[0]) & 0x3u) | ((uint32_t(idxs[1]) & 0x3u) << 2);
          uint32_t shift = static_cast<uint32_t>(pair_idx * 2); // 4 * cg
          if (fp8_k64_u32) {
            if (shift < 32)
              meta_lo |= (nibble << shift);
            else
              meta_hi |= (nibble << (shift - 32));
          } else {
            meta_val64 |= (uint64_t(nibble) << shift);
          }
        }
        if (fp8_k64_u32) {
          MetaT lo = static_cast<MetaT>(meta_lo);
          MetaT hi = static_cast<MetaT>(meta_hi);
          meta[r][strip * 2 + 0] = lo;
          meta[r][strip * 2 + 1] = hi;
          if (row_meta) {
            (*row_meta)[r * meta_cols + strip * 2 + 0] = lo;
            (*row_meta)[r * meta_cols + strip * 2 + 1] = hi;
          }
        } else {
          MetaT meta_val = static_cast<MetaT>(meta_val64);
          meta[r][strip] = meta_val;
          if (row_meta) (*row_meta)[r * strips + strip] = meta_val;
        }
      }
    }
  }

  // TODO: remove this function after all sparse utils fixed down,
  // this one is only for debug verbose purpose
  __co_host__ static inline void compress_ref(const std::vector<float>& dense_f,
                                              std::vector<float>& sparse_f,
                                              std::vector<MetaT>& meta_out,
                                              size_t M, size_t K) {
    const size_t k_sparse = K / 2;
    const bool fp8_k64_u32 =
        (META_K == 64 && std::is_same<MetaT, choreo::u32>::value);
    const size_t meta_cols = (K / META_K) * (fp8_k64_u32 ? 2 : 1);
    sparse_f.assign(M * k_sparse, 0.0f);
    meta_out.assign(M * meta_cols, MetaT(0));
    for (size_t r = 0; r < M; ++r) {
      for (size_t c_group = 0; c_group < K / 4; ++c_group) {
        size_t base = r * K + c_group * 4;
        int idxs[2] = {-1, -1};
        int nz = 0;
        for (int i = 0; i < 4; ++i) {
          if (dense_f[base + i] != 0.0f) {
            if (nz < 2) idxs[nz] = i;
            nz++;
          }
        }
        choreo_assert(nz == 2, "Invalid 2:4 structure");
        if (idxs[0] > idxs[1]) std::swap(idxs[0], idxs[1]);

        size_t sparse_col_base = c_group * 2;
        sparse_f[r * k_sparse + sparse_col_base + 0] = dense_f[base + idxs[0]];
        sparse_f[r * k_sparse + sparse_col_base + 1] = dense_f[base + idxs[1]];

        size_t pack_col = c_group / groups_per_strip;
        size_t pair_idx = (c_group % groups_per_strip) * 2;
        uint64_t nibble =
            (uint64_t(idxs[0]) & 0x3u) | ((uint64_t(idxs[1]) & 0x3u) << 2);
        if (fp8_k64_u32) {
          size_t base_col = pack_col * 2;
          uint64_t packed =
              (uint64_t(meta_out[r * meta_cols + base_col + 1]) << 32) |
              uint64_t(meta_out[r * meta_cols + base_col + 0]);
          packed |= (nibble << (pair_idx * 2));
          meta_out[r * meta_cols + base_col + 0] =
              static_cast<MetaT>(packed & 0xFFFFFFFFu);
          meta_out[r * meta_cols + base_col + 1] =
              static_cast<MetaT>((packed >> 32) & 0xFFFFFFFFu);
        } else {
          MetaT packed = meta_out[r * meta_cols + pack_col];
          packed |= (static_cast<MetaT>(idxs[0]) << (pair_idx * 2));
          packed |= (static_cast<MetaT>(idxs[1]) << ((pair_idx + 1) * 2));
          meta_out[r * meta_cols + pack_col] = packed;
        }
      }
    }
  }
};

// -----------------------------------------------------------------------------
// dtype-driven META_K inference (non-breaking addition).
// See: SparseMetaK and aliases introduced elsewhere in this file.
// -----------------------------------------------------------------------------

// Default inference: conservative default for 16-bit value types.
template <typename ValueT, typename MetaT>
struct SparseMetaK {
  static constexpr size_t value = 16;
};

// fp8 defaults (SM90 sparse MMA use wider META_K).
// METADATA K SIZE is super easy to infer
// for 2:4 sparsity, each 4 elems group has 2 non-zeros, need 2 indices with 2
// bits each to indicate its order in 0-3. thus 1 elem vs 1 bit for fp16/bf16,
// we have mma.sp shape m16n8k32 and m16n8k16 options, 32/16 is the metadata k
// size for fp8 e4m3 or e5m2, we have mma.sp shape m16n8k64, 64 is the metadata
// k size
#ifdef __CHOREO_TARGET_NATIVE_FP8_SUPPORT__
template <>
struct SparseMetaK<choreo::f8_e4m3, choreo::u32> {
  static constexpr size_t value = 64;
};
template <>
struct SparseMetaK<choreo::f8_e5m2, choreo::u32> {
  static constexpr size_t value = 64;
};
#endif

// Convenience forwarding alias (non-breaking):
template <typename ValueT, typename MetaT>
using SparseHostPolicy =
    Sparse2to4HostPolicy<ValueT, MetaT, SparseMetaK<ValueT, MetaT>::value>;

template <typename ValueT, typename MetaT = choreo::u32>
using SparsePolicy = SparseHostPolicy<ValueT, MetaT>;

// Common fixed META_K aliases (for f16/bf16 sparse MMA variants).
template <typename ValueT, typename MetaT = choreo::u32>
using SparsePolicyK16 = Sparse2to4HostPolicy<ValueT, MetaT, 16>;

template <typename ValueT, typename MetaT = choreo::u32>
using SparsePolicyK32 = Sparse2to4HostPolicy<ValueT, MetaT, 32>;

// --- Compile-time smoke tests to prevent regressions ------------------------
static_assert(SparseMetaK<choreo::f16, choreo::u32>::value == 16,
              "Regression: SparseMetaK<f16,u32> changed");
static_assert(SparseMetaK<choreo::bf16, choreo::u32>::value == 16,
              "Regression: SparseMetaK<bf16,u32> changed");
#ifdef __CHOREO_TARGET_NATIVE_FP8_SUPPORT__
static_assert(SparseMetaK<choreo::f8_e4m3, choreo::u32>::value == 64,
              "Regression: SparseMetaK<f8_e4m3,u32> changed");
static_assert(SparseMetaK<choreo::f8_e5m2, choreo::u32>::value == 64,
              "Regression: SparseMetaK<f8_e5m2,u32> changed");
#endif

static_assert(
    std::is_same<SparseHostPolicy<choreo::f16, choreo::u32>,
                 Sparse2to4HostPolicy<choreo::f16, choreo::u32, 16>>::value,
    "Regression: SparseHostPolicy<f16,u32> must match explicit instantiation");
#ifdef __CHOREO_TARGET_NATIVE_FP8_SUPPORT__
static_assert(
    std::is_same<SparseHostPolicy<choreo::f8_e4m3, choreo::u32>,
                 Sparse2to4HostPolicy<choreo::f8_e4m3, choreo::u32, 64>>::value,
    "Regression: SparseHostPolicy<f8_e4m3,u32> must match explicit "
    "instantiation");
#endif

} // namespace utils

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

// Abort if the condition/error-code is non-zero.
// Note: this intentionally takes an integer-like value (not bool) so we don't
// lose CUDA error codes via implicit conversion.
static __attribute__((always_inline)) inline void abend_true(int p) {
  if (p) {
#if defined(__CUDACC__) || defined(__CUDA__)
    auto err = static_cast<cudaError_t>(p);
    fprintf(stderr, "CUDA failure: %d (%s)\n", err, cudaGetErrorString(err));
#else
    fprintf(stderr, "Runtime failure (abend_true triggered)\n");
#endif
    std::abort();
  }
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

#ifdef __CHOREO_TARGET_CUTE__

struct TimerOption {
  int warmup = 10;
  int repeat = 100;
  bool sync = true;
};

struct ProfilerOption {
  int warmup = 10;
  int repeat = 100;
  int device = 0;
  std::string arch;
  std::string kernel_name;
  std::string ncu_path;
  std::string ncu_output = "ncu.txt";
  std::string ncu_args;
  bool page_all = true;
};

namespace detail {
inline std::string get_env(const char* key) {
  const char* val = std::getenv(key);
  return val ? std::string(val) : std::string();
}

inline bool file_exists(const std::string& path) {
  std::error_code ec;
  return !path.empty() && std::filesystem::exists(path, ec);
}

inline std::string shell_escape(const std::string& input) {
  std::string out = "'";
  for (char c : input) {
    if (c == '\'')
      out += "'\\''";
    else
      out += c;
  }
  out += "'";
  return out;
}

inline std::string sanitize_filename(const std::string& input) {
  if (input.empty()) return "kernel";
  std::string out;
  out.reserve(input.size());
  for (char c : input) {
    if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
        (c >= '0' && c <= '9') || c == '-' || c == '_')
      out.push_back(c);
    else
      out.push_back('_');
  }
  return out;
}

inline std::string resolve_ncu_path(const std::string& hint) {
  if (!hint.empty()) return hint;
  auto cuda_home = get_env("CUDA_HOME");
  if (!cuda_home.empty()) {
    auto candidate = cuda_home + "/bin/ncu";
    if (file_exists(candidate)) return candidate;
  }
  auto cuda_path = get_env("CUDA_PATH");
  if (!cuda_path.empty()) {
    auto candidate = cuda_path + "/bin/ncu";
    if (file_exists(candidate)) return candidate;
  }
  return "ncu";
}

inline std::string self_exe_path() {
  char buf[PATH_MAX];
  ssize_t len = readlink("/proc/self/exe", buf, sizeof(buf) - 1);
  if (len <= 0) return "";
  buf[len] = '\0';
  return std::string(buf);
}

inline std::string self_cmdline_tail_escaped() {
  std::ifstream fs("/proc/self/cmdline", std::ios::binary);
  if (!fs) return "";
  std::string data((std::istreambuf_iterator<char>(fs)),
                   std::istreambuf_iterator<char>());
  std::string out;
  std::string cur;
  size_t arg_idx = 0;
  for (char ch : data) {
    if (ch == '\0') {
      if (arg_idx > 0) {
        if (!out.empty()) out += " ";
        out += shell_escape(cur);
      }
      cur.clear();
      ++arg_idx;
    } else {
      cur.push_back(ch);
    }
  }
  if (!cur.empty() && arg_idx > 0) {
    if (!out.empty()) out += " ";
    out += shell_escape(cur);
  }
  return out;
}

inline std::string query_arch(int device) {
  cudaDeviceProp prop;
  cudaError_t err = cudaGetDeviceProperties(&prop, device);
  if (err != cudaSuccess) return "";
  std::ostringstream oss;
  oss << "sm" << prop.major << prop.minor;
  return oss.str();
}
} // namespace detail

template <typename F, typename... Args>
inline double timing(F&& matmul, const TimerOption& opt, Args&&... args) {
  int warmup = std::max(0, opt.warmup);
  int repeat = std::max(1, opt.repeat);

  for (int i = 0; i < warmup; ++i) matmul(std::forward<Args>(args)...);

  if (opt.sync) abend_true(cudaDeviceSynchronize());

  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  abend_true(cudaEventCreate(&start));
  abend_true(cudaEventCreate(&stop));
  abend_true(cudaEventRecord(start));
  for (int i = 0; i < repeat; ++i) matmul(std::forward<Args>(args)...);
  abend_true(cudaEventRecord(stop));
  abend_true(cudaEventSynchronize(stop));

  float elapsed_ms = 0.0f;
  abend_true(cudaEventElapsedTime(&elapsed_ms, start, stop));
  abend_true(cudaEventDestroy(start));
  abend_true(cudaEventDestroy(stop));
  return static_cast<double>(elapsed_ms) / static_cast<double>(repeat);
}

template <typename F, typename... Args>
inline bool profile(F&& matmul, ProfilerOption& opt, Args&&... args) {
  if (opt.device >= 0) abend_true(cudaSetDevice(opt.device));
  if (opt.arch.empty()) opt.arch = detail::query_arch(opt.device);

  const char* in_run = std::getenv("CHOREO_PROFILE_RUN");
  if (in_run && std::string(in_run) == "1") {
    int warmup = std::max(0, opt.warmup);
    int repeat = std::max(1, opt.repeat);
    for (int i = 0; i < warmup; ++i) matmul(std::forward<Args>(args)...);
    for (int i = 0; i < repeat; ++i) matmul(std::forward<Args>(args)...);
    abend_true(cudaDeviceSynchronize());
    return true;
  }

  std::string ncu = detail::resolve_ncu_path(opt.ncu_path);
  opt.ncu_path = ncu;
  std::string exe = detail::self_exe_path();
  if (exe.empty()) {
    std::cerr << "[choreo] failed to resolve current executable path.\n";
    return false;
  }

  std::string cmd_args = detail::self_cmdline_tail_escaped();
  std::string out_file = opt.ncu_output;
  if (out_file.empty() || out_file == "ncu.txt") {
    std::string kname = detail::sanitize_filename(opt.kernel_name);
    out_file = kname + ".ncu.txt";
  }
  opt.ncu_output = out_file;

  std::ostringstream cmd;
  cmd << "CHOREO_PROFILE_RUN=1 ";
  if (opt.device >= 0) cmd << "CUDA_VISIBLE_DEVICES=" << opt.device << " ";
  cmd << "sudo -E " << detail::shell_escape(ncu) << " ";
  if (!opt.ncu_args.empty()) cmd << opt.ncu_args << " ";
  cmd << detail::shell_escape(exe) << " ";
  if (!cmd_args.empty()) cmd << cmd_args << " ";
  cmd << "> " << detail::shell_escape(out_file) << " 2>&1";

  int ret = std::system(cmd.str().c_str());
  return ret == 0;
}

#endif // __CHOREO_TARGET_CUTE__

// target specific definations
#ifdef __CHOREO_PRIVATE_TGT0__
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

} // end namespace choreo

// Expose sub-byte float aliases in global namespace for generated device code
#ifdef __CHOREO_TARGET_NATIVE_FP6_SUPPORT__
using choreo::f6_e2m3;
using choreo::f6_e3m2;
#endif
#ifdef __CHOREO_TARGET_NATIVE_FP4_SUPPORT__
using choreo::f4_e2m1;
#endif

// target specific libraries (non-shared)
#if __has_include("choreo_cute.h")
  #include "choreo_cute.h"
#endif

#if __has_include("private_target0_runtime.h")
  #include "private_target0_runtime.h"
#endif

namespace choreo {

#if defined(__CHOREO_PRIVATE_TGT0__) || defined(__CHOREO_TARGET_CUTE__)

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
