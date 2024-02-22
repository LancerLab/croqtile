#ifndef __CHOREO_H__
#define __CHOREO_H__

#if __cplusplus < 201703L
  #error "Choreo requires C++17 or later"
#endif

#include <array>
#include <cstdint> // For fixed-width integer types

namespace choreo {

template<int N> using mdspan = std::array<N>;

template <typename T, int N>
struct dataspan {
  T* data;
  const std::array<int, N> span;
  dataspan(T* d, const std::array<int, N> & s) : data(d), span(s) {}
};

// Floating-point types
using f32 = float;
using f16 = float16;

// Check for __bf16 support
#if defined(__clang__)
	#if __clang_major__ >= 11
		#define BF16_SUPPORTED 1
		using bf16 = __bf16;
	#endif
#elif defined(__GNUC__)
	#if __GNUC__ >= 11
		#define BF16_SUPPORTED 1
		using bf16 = __bf16;
	#endif
#endif

#ifndef BF16_SUPPORTED
	#error "Compiler does not support __bf16. Please use a compiler that supports __bf16 or define a fallback type."
#endif

// Unsigned integer types
using u32 = uint32_t; // 32-bit unsigned integer
using u16 = uint16_t; // 16-bit unsigned integer
using u8 = uint8_t;  // 8-bit unsigned integer

// Signed integer types
using s32 = int32_t; // 32-bit signed integer
using s16 = int16_t; // 16-bit signed integer
using s8 = int8_t;   // 8-bit signed integer

} // end namespace choreo

#endif // __CHOREO_H__
