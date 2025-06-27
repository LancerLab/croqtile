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
#else
#define __co_device__
#define __co_host__
#define __co_any__
#endif // __TOPSCC__

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
using f32 = float;

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
using f16 = __fp16;
using half = __fp16;
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

using bf16 = __bf16;
using bfp16 = __bf16;
using bfloat16 = __bf16;

// Check for __bf16 support
#if !defined(__TOPSCC__) || !defined(__clang__) || !defined(__GNUC__)
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

namespace utils {

// template <typename U>
// inline void fill_random(U*, size_t, U, U);

// specializations

// f32
template <typename U>
inline typename std::enable_if<std::is_same<U, float>::value, void>::type
fill_random(U* array, size_t N, U lb, U ub) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<U> rand_func(lb,
                                              ub); // [-1.0, 1.0)

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

// s32/u32 ...
// if T is integer，utilize std::uniform_int_distribution
template <typename U>
inline typename std::enable_if<std::is_integral<U>::value, void>::type
fill_random(U* array, size_t N, U lb, U ub) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<U> rand_func(lb, ub); // [-100, 100]

  std::generate_n(&array[0], N, [&]() { return rand_func(gen); });
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

  __co_any__ void fill(T value) {
    fill_n(this->data(), this->element_count(), value);
  }

  __co_host__ void fill_random(T lb, T ub) {
    utils::fill_random(this->data(), this->element_count(), lb, ub);
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

  __co_host__ void fill(T value) {
    fill_n(data(), this->element_count(), value);
  }

  __co_host__ void fill_random(T lb, T ub) {
    utils::fill_random(data(), element_count(), lb, ub);
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
public:
  struct Chunk {
    size_t size;
    size_t start_time;
    size_t end_time;
    std::string buffer_id;
  };
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

    size_t size = chunks.size();

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
        size, std::vector<bool>(size, false));

    for (size_t i = 0; i < size; ++i) {
      for (size_t j = i + 1; j < size; ++j) {
        if (sorted_chunks[i].start_time <= sorted_chunks[j].end_time &&
            sorted_chunks[j].start_time <= sorted_chunks[i].end_time) {
          interference_graph[i][j] = true;
          interference_graph[j][i] = true;
        }
      }
    }

    // assign space for each buffer
    std::map<size_t, size_t> assigned_offsets;

    using Range = std::pair<size_t, size_t>;

    for (size_t i = 0; i < size; ++i) {
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
        result.chunk_offsets[chunk.buffer_id] = -1;
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

// target specific definations
#ifdef __TOPSCC__
template <typename T>
__device__ static int inline __addr2int__(T* v) {
  return static_cast<int>(reinterpret_cast<long long>(v));
}
#else
template <typename T>
static int inline __addr2int__(T* v) {
  return (int)v;
}
#endif

#ifdef __TOPSCC__

} // end namespace choreo

#include <krt/builtins.h>

namespace choreo {

// For tops API check: abend on failures
static __attribute__((always_inline)) inline void abend_false(bool p) {
  if (!p) std::abort();
}

static __attribute__((always_inline)) inline void abend_true(bool p) {
  if (p) std::abort();
}

// --- light-weight choreo-topscc device library --- //

__device__ __attribute__((always_inline)) static inline void __co_abort__() {
#ifdef __CHOREO_USE_TOPS_ABORT__
  tops::abort();
#else
  abort();
#endif
}

// choreo device future
struct future {
  tops_dte_ctx_t* ctx = nullptr;
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

  __device__ future(tops_dte_ctx_t& dte, const char* n, unsigned l, unsigned c,
                    void* data = nullptr)
      : ctx(&dte), d(data), s(ST_NONE), name(n), line(l), column(c) {}

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
