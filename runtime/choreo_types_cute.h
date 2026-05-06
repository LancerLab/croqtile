#ifndef __CHOREO_TYPES_CUTE_H__
#define __CHOREO_TYPES_CUTE_H__

// CuTe/CUDA-specific type operations: FP8 arithmetic operators, bitcast
// utilities.
//
// This header is conditionally included by choreo.h when targeting CUDA/CuTe.
// It requires choreo_types.h to have been included first.

#ifdef __CHOREO_TARGET_CUTE__

namespace choreo {

  // Minimal arithmetic support for FP8 scalar types.
  // Choreo's codegen may form expressions like `fp8 + fp8` before casting.
  // CUTLASS/CUTE FP8 types and CUDA FP8 types don't consistently provide these
  // operators, so we define them here and return FP32.
  #ifdef __CHOREO_TARGET_NATIVE_FP8_SUPPORT__
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
  #endif   // __CHOREO_TARGET_NATIVE_FP8_SUPPORT__

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
    #if !defined(__CHOREO_TARGET_NATIVE_TF32_SUPPORT__) ||                     \
        !defined(__CHOREO_TARGET_NATIVE_F16_SUPPORT__) ||                      \
        !defined(__CHOREO_TARGET_NATIVE_BF16_SUPPORT__) ||                     \
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

} // namespace choreo

#endif // __CHOREO_TARGET_CUTE__

#endif // __CHOREO_TYPES_CUTE_H__
