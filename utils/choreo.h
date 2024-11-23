#ifndef __CHOREO_H__
#define __CHOREO_H__

#if __cplusplus < 201703L
// #error "Choreo requires C++17 or later"
#endif

#include <algorithm>
#include <cstdint>          // For fixed-width integer types
#include <initializer_list> // for std::initializer_list
#include <iostream>         // report error
#include <memory>
#include <random>

namespace choreo {

inline void choreo_assert(bool p, const char* msg, const char* file = __FILE__,
                          int line = __LINE__) {
  if (!p) {
    std::cerr << "Assertion failed: " << msg << ", file " << file << ", line "
              << line << std::endl;
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

namespace {

template <typename T, size_t N>
class SimpleArray {
  static_assert(N > 0, "can not create 0-dim array");

public:
  // Constructor for brace-initialization
  SimpleArray(std::initializer_list<T> init) {
    std::size_t count = 0;
    for (auto& value : init) {
      if (count >= N) break; // Avoid exceeding the array size
      data[count++] = value;
    }
  }

  SimpleArray(const SimpleArray&) = default;
  SimpleArray& operator=(const SimpleArray&) = default;
  ~SimpleArray() = default;

  // Returns the element at specified index
  T& operator[](uint32_t index) { return data[index]; }

  // Returns the element at specified index (const version)
  const T& operator[](uint32_t index) const { return data[index]; }

  // Returns the number of elements in the array
  constexpr uint32_t size() const noexcept { return N; }

  // Returns a pointer to the underlying array serving as element storage
  T* begin() { return data; }
  const T* begin() const { return data; }

  T* end() { return data + N; }
  const T* end() const { return data + N; }

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
inline static bool operator==(const SimpleArray<T, N>& l,
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
  ArrayProxy(T* arr, const mdspan<N>& dimensions, size_t off)
      : data(arr), dims(&dimensions), offset(off) {}

  template <size_t M = N>
  typename std::enable_if<(M == 1),
                          T&>::type // make sure to return the reference type
  operator[](int index) {
    choreo_assert(index >= 0, "Index out of bounds", __FILE__, __LINE__);
    choreo_assert((size_t)index < (*dims)[0], "Index out of bounds", __FILE__,
                  __LINE__);

    // Direct element access
    return data[offset + (size_t)index];
  }

  template <size_t M = N>
  typename std::enable_if<(M > 1), ArrayProxy<T, N - 1>>::type
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

#ifndef NATIVE_F16_SUPPORT
// this f16 accepts literal initialization, but without arith support
class f16 {
private:
  uint16_t bits; // Storage for the half-precision bits

public:
  // Default constructor
  f16() : bits(0) {}

  // Constructor for conversion from float
  f16(float value) { bits = floatToHalfBits(value); }

  // Constructor for conversion from double
  f16(double value) { bits = floatToHalfBits(static_cast<float>(value)); }

  // Implicit conversion from float
  f16& operator=(float value) {
    bits = floatToHalfBits(value);
    return *this;
  }

  // Implicit conversion from double
  f16& operator=(double value) {
    bits = floatToHalfBits(static_cast<float>(value));
    return *this;
  }

  bool operator==(f16 value) {
    return halfBitsToFloat(bits) == value.toFloat();
  }

  // Function to convert float to half precision bits (naive and placeholder)
  static uint16_t floatToHalfBits(float value) {
    // Simplified conversion: this does not handle rounding, infinities, or NaNs
    // correctly In practice, use a library or a fully implemented conversion
    // function
    int32_t fltInt32 = *((int32_t*)&value);
    int32_t t1 = (fltInt32 & 0x7FFFFFFF) >> 13; // Non-sign bits
    int32_t t2 = (fltInt32 & 0x80000000) >> 16; // Sign bit
    int32_t t3 = ((fltInt32 & 0x7F800000) >> 13) - (112 << 10);

    int32_t t4 = std::max(0, std::min(t3, (1 << 10) - 1));
    return (t2 | t4 | t1);
  }

  // Function to convert half precision bits to float (naive and placeholder)
  static float halfBitsToFloat(uint16_t bits) {
    // Simplified conversion: this does not handle rounding, infinities, or NaNs
    // correctly In practice, use a library or a fully implemented conversion
    // function
    int32_t t1 = (bits & 0x7FFF) << 13; // Non-sign bits
    int32_t t2 = (bits & 0x8000) << 16; // Sign bit
    int32_t t3 = ((bits & 0x7C00) << 13) + (112 << 23);

    int32_t fltInt32 = t2 | t3 | t1;
    return *((float*)&fltInt32);
  }

  // Method to get the float value from the f16 object
  float toFloat() const { return halfBitsToFloat(bits); }
};
#else
using f16 = __fp16;
#endif // NATIVE_F16_SUPPORT

#ifndef NATIVE_BF16_SUPPORT
class bf16 {
private:
  uint16_t bits; // Storage for the half-precision bits

public:
  // Default constructor
  bf16() : bits(0) {}

  // Constructor for conversion from float
  bf16(float value) { bits = floatToHalfBits(value); }

  // Constructor for conversion from double
  bf16(double value) { bits = floatToHalfBits(static_cast<float>(value)); }

  // Implicit conversion from float
  bf16& operator=(float value) {
    bits = floatToHalfBits(value);
    return *this;
  }

  // Implicit conversion from double
  bf16& operator=(double value) {
    bits = floatToHalfBits(static_cast<float>(value));
    return *this;
  }

  bool operator==(bf16 value) {
    return halfBitsToFloat(bits) == value.toFloat();
  }

  // Function to convert float to half precision bits (naive and placeholder)
  static uint16_t floatToHalfBits(float value) {
    // Simplified conversion: this does not handle rounding, infinities, or NaNs
    // correctly In practice, use a library or a fully implemented conversion
    // function
    int32_t fltInt32 = *((int32_t*)&value);
    return (fltInt32 & 0xFFFF0000) >> 16;
  }

  // Function to convert half precision bits to float (naive and placeholder)
  static float halfBitsToFloat(uint16_t bits) {
    int32_t fltInt32 = ((uint32_t)bits) << 16;
    return *((float*)&fltInt32);
  }

  // Method to get the float value from the bf16 object
  float toFloat() const { return halfBitsToFloat(bits); }
};
#else
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
#endif // NATIVE_BF16_SUPPORT

#ifndef BF16_SUPPORTED
//#error \
//    "Compiler does not support __bf16. Please use a compiler that supports __bf16 or define a fallback type."
#endif

// Unsigned integer types
using u32 = uint32_t; // 32-bit unsigned integer
using u16 = uint16_t; // 16-bit unsigned integer
using u8 = uint8_t;   // 8-bit unsigned integer

// Signed integer types
using s32 = int32_t; // 32-bit signed integer
using s16 = int16_t; // 16-bit signed integer
using s8 = int8_t;   // 8-bit signed integer

// A 'spanned_view' is a memview of data. It is ranked, but no necessary to have
// compile-time dimensions
template <typename T, size_t Rank>
class spanned_view {
  static_assert(Rank != 0, "unexpected 0-dims.");
  T* ptr = nullptr;
  const mdspan<Rank> dims;

public:
  explicit spanned_view(T* d, const mdspan<Rank>& s) : ptr(d), dims(s) {}

  constexpr size_t rank() const { return Rank; }
  const mdspan<Rank>& shape() const { return dims; }

  size_t size() const { return span_size(dims); }
  size_t bytes() const { return size() * sizeof(T); }
  T* data() { return ptr; }
  T* data() const { return ptr; }

  // allow multi-dim-style access, be like: a[1][3]
  template <size_t M = Rank>
  typename std::enable_if<(M == 1),
                          T&>::type // make sure to return the reference type
  operator[](int index) {
    choreo_assert(index >= 0, "Index out of bounds", __FILE__, __LINE__);
    choreo_assert((size_t)index < dims[0], "Index out of bounds", __FILE__,
                  __LINE__);
    return ptr[index];
  }

  template <size_t M = Rank>
  typename std::enable_if<(M > 1), ArrayProxy<T, Rank - 1>>::type
  operator[](int index) {
    choreo_assert(index >= 0, "Index out of bounds", __FILE__, __LINE__);
    choreo_assert((size_t)index < dims[0], "Index out of bounds", __FILE__,
                  __LINE__);
    const auto& sub_dims =
        *reinterpret_cast<const mdspan<Rank - 1>*>(&(dims[1]));
    return ArrayProxy<T, Rank - 1>(ptr, sub_dims, (size_t)index * dims[1]);
  }

  friend bool operator==(const spanned_view& l, const spanned_view& r) {
    if (l.dims != r.dims) return false;

    for (size_t i = 0; i < l.size(); ++i)
      if (l.ptr[i] != r.ptr[i]) return false;

    return true;
  }

  void fill_random(T lb, T ub) {
    fill_random(this->data(), this->size(), lb, ub);
  }

  template <typename U>
  typename std::enable_if<std::is_same<U, f16>::value ||
                          std::is_same<U, bf16>::value>::type
  fill_random(float lb, float ub) {
    fill_random(this->data(), this->size(), lb, ub);
  }

private:
  // f32
  template <typename U>
  typename std::enable_if<std::is_same<U, float>::value>::type
  fill_random(U* array, size_t N, U lb, U ub) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<U> rand_func(lb,
                                                ub); // 浮点数范围 [-1.0, 1.0)

    std::generate_n(&array[0], N, [&]() { return rand_func(gen); });
  }

  // f16/bf16
  template <typename U>
  typename std::enable_if<std::is_same<U, f16>::value ||
                          std::is_same<U, bf16>::value>::type
  fill_random(U* array, size_t N, U lb, U ub) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> rand_func(
        lb.toFloat(),
        ub.toFloat()); // 浮点数范围 [-1.0, 1.0)

    std::generate_n(&array[0], N, [&]() { return U(rand_func(gen)); });
  }

  // f16/bf16 with float lb/ub
  template <typename U>
  typename std::enable_if<std::is_same<U, f16>::value ||
                          std::is_same<U, bf16>::value>::type
  fill_random(U* array, size_t N, float lb, float ub) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> rand_func(lb, ub);

    std::generate_n(&array[0], N, [&]() { return U(rand_func(gen)); });
  }

  // s32/u32 ...
  // 如果 T 是整数类型，使用 std::uniform_int_distribution
  template <typename U>
  typename std::enable_if<std::is_integral<U>::value>::type
  fill_random(U* array, size_t N, U lb, U ub) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<U> rand_func(lb, ub); // 整数范围 [-100, 100]

    std::generate_n(&array[0], N, [&]() { return rand_func(gen); });
  }
};

// A 'spanned_data' is similar to 'spanned_view' but manage memory
template <typename T, size_t Rank>
class spanned_data {
  std::unique_ptr<T[]> ptr = nullptr; // this is used as the output
  mdspan<Rank> dims;

public:
  explicit spanned_data(std::unique_ptr<T[]>&& d, const mdspan<Rank>& s)
      : ptr(std::move(d)), dims(s) {}

  spanned_data(const spanned_data&) = delete; // move only
  spanned_data& operator=(const spanned_data&) = delete;

  spanned_data(spanned_data&& sd) : ptr(std::move(sd.ptr)), dims(sd.dims) {}

  constexpr size_t rank() const { return Rank; }
  const mdspan<Rank>& shape() const { return dims; }

  size_t size() const { return span_size(dims); }
  size_t bytes() const { return size() * sizeof(T); }
  T* data() { return ptr.get(); }

  // allow multi-dim-style access, be like: a[1][3]
  template <size_t M = Rank>
  typename std::enable_if<(M == 1),
                          T&>::type // make sure to return the reference type
  operator[](int index) {
    choreo_assert(index >= 0, "Index out of bounds", __FILE__, __LINE__);
    choreo_assert((size_t)index < dims[0], "Index out of bounds", __FILE__,
                  __LINE__);
    return ptr[index];
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

    for (size_t i = 0; i < l.size(); ++i)
      if (l.ptr[i] != r.ptr[i]) return false;

    return true;
  }
};

template <size_t Rank>
mdspan<Rank> make_mdspan(const std::initializer_list<size_t>& init) {
  return mdspan<Rank>(init);
}

// note: spanned_view does not invoke copy. Instead, it associates data with a
// multi-dimension view of memory
template <size_t Rank, typename T>
spanned_view<T, Rank> make_spanview(T* ptr,
                                    std::initializer_list<size_t> init) {
  return spanned_view<T, Rank>(ptr, make_mdspan<Rank>(init));
}

template <typename T, size_t N, size_t M>
spanned_view<T, 2> make_spanview(T (&arr)[N][M]) {
  return spanned_view<T, 2>((T*)arr, {N, M});
}

template <typename T, size_t Rank>
spanned_data<T, Rank> make_spandata(std::initializer_list<size_t> init) {
  size_t size = 1;
  for (auto& value : init) size *= value;
  choreo_assert(size > 1, "error: invalid size.", __FILE__, __LINE__);

  return spanned_data<T, Rank>(std::make_unique<T[]>(size),
                               make_mdspan<Rank>(init));
}

// converting from vector to another type
template <size_t Rank, typename T>
auto copy_as_spanned(T* ptr, std::initializer_list<size_t> init) {
  size_t size = 1;
  for (auto& value : init) size *= value;
  choreo_assert(size > 1, "error: invalid size.", __FILE__, __LINE__);

  auto parr = std::make_unique<T[]>(size);
  std::copy(ptr, ptr + size, parr.get());
  auto res = spanned_data<T, Rank>(std::move(parr), make_mdspan<Rank>(init));
  choreo_assert(res.bytes() == size * sizeof(T), "error: size does not match.",
                __FILE__, __LINE__);
  return res;
}

} // end namespace choreo

#endif // __CHOREO_H__
