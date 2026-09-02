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

namespace detail {

template <typename To, typename From>
__co_any__ inline static To bitCopy(const From& value) {
  static_assert(sizeof(To) == sizeof(From),
                "bit copy requires identically sized types");
  To result;
  __builtin_memcpy(&result, &value, sizeof(To));
  return result;
}

} // namespace detail

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

  // IEEE-754 round-to-nearest-even conversion float32 -> float16, used on
  // targets where half is this soft struct (no native/backend half type whose
  // own RNE narrowing would supersede it).
  uint32_t x = detail::bitCopy<uint32_t>(value);
  uint32_t sign = (x >> 16) & 0x8000u;
  uint32_t exp = (x >> 23) & 0xFFu;
  uint32_t man = x & 0x7FFFFFu;
  uint16_t bits;

  if (exp == 0xFFu) { // Inf / NaN: pass exponent through, keep NaN payload
    if (man == 0u)
      bits = (uint16_t)(sign | 0x7C00u);
    else {
      uint32_t hm = man >> 13;
      if (hm == 0u) hm = 1u;
      bits = (uint16_t)(sign | 0x7C00u | hm);
    }
    return detail::bitCopy<T>(bits);
  }
  if (exp == 0u) { // float32 zero or subnormal.  All float32 subnormals are
    // < 2^-24, below the smallest float16 subnormal, so they flush to +/-0.
    bits = (uint16_t)sign;
    return detail::bitCopy<T>(bits);
  }

  int e = (int)exp - 127; // unbiased exponent
  if (e > 15) {           // overflow -> +/-inf
    bits = (uint16_t)(sign | 0x7C00u);
    return detail::bitCopy<T>(bits);
  }
  if (e >= -14) { // normal float16
    uint32_t halfExp = (uint32_t)(e + 15);
    uint32_t hm = man >> 13;
    uint32_t rem = man & 0x1FFFu;
    if (rem > 0x1000u || (rem == 0x1000u && (hm & 1u))) ++hm; // RNE
    if (hm == 0x400u) {
      hm = 0u;
      ++halfExp;
    }                       // mantissa overflow carry
    if (halfExp >= 0x1Fu) { // rounded up past max finite -> inf
      bits = (uint16_t)(sign | 0x7C00u);
      return detail::bitCopy<T>(bits);
    }
    bits = (uint16_t)(sign | (halfExp << 10) | hm);
    return detail::bitCopy<T>(bits);
  }

  // Values below half the smallest float16 subnormal round to signed zero.
  // e == -25 is retained so values immediately above the exact tie round up.
  if (e < -25) {
    bits = (uint16_t)sign;
    return detail::bitCopy<T>(bits);
  }

  // e in [-25, -15]: float16 subnormal.  value = (2^23+man) * 2^(e-23); the
  // float16 subnormal field holds frac with value = frac * 2^-24, so
  // frac = round((2^23+man) * 2^(e+1)).
  {
    uint32_t sh = (uint32_t)(-(e + 1)); // 14..24
    uint32_t m = (1u << 23) | man;
    uint32_t frac = m >> sh;
    uint32_t rem = m & ((1u << sh) - 1u);
    if (rem > (1u << (sh - 1)) || (rem == (1u << (sh - 1)) && (frac & 1u)))
      ++frac; // RNE
    bits = (uint16_t)(sign | (frac >= 0x400u ? 0x0400u : frac));
    return detail::bitCopy<T>(bits);
  }
}

// Function to convert half precision bits to float
// Refer to https://en.wikipedia.org/wiki/Half-precision_floating-point_format
//    and https://en.wikipedia.org/wiki/Single-precision_floating-point_format
template <typename T, typename F>
__co_any__ inline static T __f16_to_f32(F value) {
  static_assert(sizeof(T) == 4, "target is not a float.");
  static_assert(sizeof(F) == 2, "source is not a half float.");

  uint16_t h = detail::bitCopy<uint16_t>(value);
  uint32_t sign = ((uint32_t)h & 0x8000u) << 16;
  uint32_t exp = (h >> 10) & 0x1Fu; // 5-bit exponent
  uint32_t frac = h & 0x3FFu;       // 10-bit fraction
  uint32_t resultBits;

  if (exp == 0x1Fu) { // +/-inf or NaN
    if (frac == 0u)
      resultBits = sign | 0x7F800000u; // +/-inf (was wrongly decoded as 0)
    else
      resultBits = sign | 0x7F800000u | (frac << 13); // NaN, payload preserved
  } else if (exp == 0u) {
    if (frac == 0u)
      resultBits = sign; // +/-0
    else { // subnormal: value = frac * 2^-24, normalise into float32
      uint32_t j =
          31u - (uint32_t)__builtin_clz((uint32_t)frac); // top bit 0..9
      resultBits =
          sign | ((103u + j) << 23) | ((frac - (1u << j)) << (23u - j));
    }
  } else { // normal: value = (1 + frac/2^10) * 2^(exp-15); 112 = 127 - 15
    resultBits = sign | ((exp + 112u) << 23) | (frac << 13);
  }
  return detail::bitCopy<T>(resultBits);
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

using half = f16; // soft host/device fallback: an f16 value type, NOT a bare
                  // unsigned short (avoids silent integer arithmetic misuse)

inline std::ostream& operator<<(std::ostream& os, const f16& v) {
  os << (float)v;
  return os;
}

#else
  #ifndef __CHOREO_F16_DEFINED__
    // __CHOREO_F16_DEFINED__ is set by the private accelerator-backend header,
    // where choreo::f16/half alias the SDK half storage struct (value semantics
    // + RNE).  Everywhere else f16/half are the native builtin / backend half
    // type below.
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
  #endif
#endif // __CHOREO_TARGET_NATIVE_F16_SUPPORT__

#ifndef __CHOREO_F16_CONVERT_DEFINED__
// Where __CHOREO_F16_CONVERT_DEFINED__ is set (private accelerator-backend
// header aliasing choreo::f16 to the SDK half storage struct) these narrow /
// widen via that struct's RNE; everywhere else they resolve to the soft
// conversions below.
__co_any__ inline static f16 f32_to_f16(f32 value) {
  #if !defined(__CHOREO_TARGET_NATIVE_F16_SUPPORT__)
  return f16(value);
  #elif defined(__USE_CUDA_TYPE__)
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
#endif // __CHOREO_F16_CONVERT_DEFINED__

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

  // Function to convert float to bfloat16 bits (IEEE-754 round-to-nearest-even
  // on the low 16 dropped mantissa bits).  Host/soft fallback only: where
  // choreo::bf16 aliases the SDK bfloat16 storage struct, that struct's RNE
  // narrowing supersedes this.
  __co_any__ static uint16_t floatToHalfBits(float value) {
    uint32_t fltInt32 = detail::bitCopy<uint32_t>(value);
    uint32_t sign = fltInt32 >> 31;
    uint32_t exp = (fltInt32 >> 23) & 0xFF;
    uint32_t man = fltInt32 & 0x7FFFFF;
    if (exp == 0xFFu) { // Inf / NaN: keep exponent, preserve a NaN payload
      if (man == 0u) return (uint16_t)((sign << 15) | 0x7F80u);
      uint16_t hm = (uint16_t)(man >> 16);
      if ((hm & 0x7Fu) == 0u) hm |= 0x40u;
      return (uint16_t)((sign << 15) | 0x7F80u | (hm & 0x7Fu));
    }
    uint16_t res = (uint16_t)(fltInt32 >> 16); // sign|exp|top-7 mantissa bits
    uint32_t rem = fltInt32 & 0xFFFFu;
    if (rem > 0x8000u || (rem == 0x8000u && (res & 1u))) ++res; // RNE
    return res; // carries roll into exponent; exp 0xFF result = inf (correct)
  }

  // Function to convert bfloat16 bits to float (always exact for finite
  // values: the sign and 8-bit exponent fields already align with float32, so
  // only the 7-bit mantissa is left-aligned into the 23-bit field).
  __co_any__ static float halfBitsToFloat(uint16_t bits) {
    uint32_t fltInt32 = ((uint32_t)bits) << 16;
    return detail::bitCopy<float>(fltInt32);
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
