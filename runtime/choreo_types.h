#ifndef __CHOREO_TYPES_H__
#define __CHOREO_TYPES_H__

// Choreo scalar type definitions and conversion utilities.
//
// This header is included by choreo.h after target-detection macros
// (__CHOREO_TARGET_NATIVE_F16_SUPPORT__, __CHOREO_TARGET_NATIVE_BF16_SUPPORT__,
// etc.) have been defined.  It must NOT be included standalone.

namespace choreo {

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
  #elif defined(__CHOREO_TARGET_AMDGPU__)
using f16 = ::__half;
using half = ::__half;
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

  #ifndef __CHOREO_BF16_DEFINED__
    #ifdef __CHOREO_TARGET_CUTE__
      #ifdef __USE_CUTE_TYPE__
using __bf16 = cute::bfloat16_t;
      #else
using __bf16 = __nv_bfloat16;
      #endif
using bf16 = __bf16;
using bfp16 = __bf16;
using bfloat16 = __bf16;
    #elif defined(__CHOREO_TARGET_AMDGPU__)
using bf16 = ::hip_bfloat16;
using bfp16 = ::hip_bfloat16;
using bfloat16 = ::hip_bfloat16;
    #else
using bf16 = __bf16;
using bfp16 = __bf16;
using bfloat16 = __bf16;
    #endif
  #endif // __CHOREO_BF16_DEFINED__

  #if !defined(__CHOREO_PRIVATE_TGT0__) && !defined(__clang__) &&              \
      !defined(__GNUC__) && !defined(__CUDACC__)
    #error                                                                     \
        "Compiler does not support __bf16. Please use a compiler that supports __bf16 or define a fallback type."
  #endif

#endif // __CHOREO_TARGET_NATIVE_BF16_SUPPORT__

#ifndef __CHOREO_BF16_CONVERT_DEFINED__
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
  return static_cast<f32>(value);
  #endif
}
#endif // __CHOREO_BF16_CONVERT_DEFINED__

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
    return f16_to_f32(value);
  } else if constexpr (std::is_same<T, bf16>::value) {
    return bf16_to_f32(value);
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
    static_assert(sizeof(T) == 0, "Unsupported type for to_f32 conversion.");
  }
}

namespace utils {
template <typename U>
__co_any__ inline U from_f32(float v) {
  if constexpr (std::is_same<U, f16>::value) {
    return f32_to_f16(v);
  } else if constexpr (std::is_same<U, bf16>::value) {
    return f32_to_bf16(v);
#ifdef __CHOREO_TARGET_NATIVE_FP8_SUPPORT__
  } else if constexpr (std::is_same<U, f8_e4m3>::value) {
    return f8_e4m3(v);
  } else if constexpr (std::is_same<U, f8_e5m2>::value) {
    return f8_e5m2(v);
#endif
  } else {
    return static_cast<U>(v);
  }
}
} // namespace utils

} // namespace choreo

#endif // __CHOREO_TYPES_H__
