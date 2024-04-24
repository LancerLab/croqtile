#ifndef __CHOREO_H__
#define __CHOREO_H__

#if __cplusplus < 201703L
#error "Choreo requires C++17 or later"
#endif

#include <cstdint>           // For fixed-width integer types
#include <initializer_list>  // for std::initializer_list
#include <iostream>          // report error

namespace choreo {

[[noreturn]] inline void choreo_assert(bool p, const char* msg,
                                       const char* file = __FILE__,
                                       int line = __LINE__) {
  if (!p) {
    std::cerr << "Assertion failed: " << msg << ", file " << file << ", line "
              << line << std::endl;
    std::abort();
  }
}

namespace {
template <typename T, uint32_t N>
class SimpleArray {
 public:
  // Constructor for brace-initialization
  SimpleArray(std::initializer_list<T> init) {
    std::size_t count = 0;
    for (auto& value : init) {
      if (count >= N) break;  // Avoid exceeding the array size
      data[count++] = value;
    }
  }

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

 private:
  T data[N];
};

}  // end anonymous namespace

template <int Rank>
using mdspan = SimpleArray<int, Rank>;

// A spanned data is ranked, but no necessary to have compile-time dimensions
template <typename T, int Rank>
struct spanned {
  T* data = nullptr;
  const mdspan<Rank> shape;
  explicit spanned(T* d, const mdspan<Rank>& s) : data(d), shape(s) {}

  size_t dims() const {
    choreo_assert(shape.size() == 0, "unexpected size == 0");
    return shape.size();
  }

  size_t size() const {
    choreo_assert(shape.size() == 0, "unexpected size == 0");
    unsigned sz = 1;
    for (auto itr = shape.begin(); itr != shape.end(); ++itr) sz *= *itr;
    return sz;
  }

  size_t bytes() const { return size() * sizeof(T) }
};

template <int Rank>
mdspan<Rank> make_mdspan(std::initializer_list<int> init) {
  return mdspan<Rank>(init);
}

// note: spanned does not invoke copy. Instead, it associates data with a
// multi-dimension view of memory
template <int Rank, typename T>
spanned<T, Rank> make_spanned(T* ptr, std::initializer_list<int> init) {
  return spanned<T, Rank>(ptr, make_mdspan<Rank>(init));
}

template <typename T, int N, int M>
spanned<T, 2> make_spanned(T (&arr)[N][M]) {
  return spanned<T, 2>((T*)arr, {N, M});
}

// converting from vector of another type
template <int Rank, typename T, typename U>
spanned<T, Rank> make_spanned(const std::vector<U>& d,
                              std::initializer_list<int> init) {
  auto res = make_spanned<Rank>((T*)d.data(), init);
  choreo_assert(res.bytes() == d.size() * sizeof(U), "size does not match");
  return res;
}

// Floating-point types
using f32 = float;
using f16 = __fp16;

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
//#error \
    "Compiler does not support __bf16. Please use a compiler that supports __bf16 or define a fallback type."
#endif

// Unsigned integer types
using u32 = uint32_t;  // 32-bit unsigned integer
using u16 = uint16_t;  // 16-bit unsigned integer
using u8 = uint8_t;    // 8-bit unsigned integer

// Signed integer types
using s32 = int32_t;  // 32-bit signed integer
using s16 = int16_t;  // 16-bit signed integer
using s8 = int8_t;    // 8-bit signed integer

}  // end namespace choreo

#endif  // __CHOREO_H__
