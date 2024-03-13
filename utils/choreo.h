#ifndef __CHOREO_H__
#define __CHOREO_H__

#if __cplusplus < 201703L
#error "Choreo requires C++17 or later"
#endif

#include <cstdint>           // For fixed-width integer types
#include <initializer_list>  // for std::initializer_list

namespace choreo {

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

template <int N>
using mdspan = SimpleArray<int, N>;

template <typename T, int N>
struct spanned {
  T* data;
  const mdspan<N> span;
  explicit spanned(T* d, const mdspan<N>& s) : data(d), span(s) {}
  unsigned size() {
    if (span.size() == 0) return 0;

    unsigned sz = 1;
    for (auto itr = span.begin(); itr != span.end(); ++itr)
      sz *= *itr;
    return sz;
  }
};

template <int N>
mdspan<N> make_mdspan(std::initializer_list<int> init) {
  return mdspan<N>(init);
}

template <int N, typename T>
spanned<T, N> make_spanned(T* ptr, std::initializer_list<int> init) {
  return spanned<T, N>(ptr, make_mdspan<N>(init));
}

template <typename T, int N, int M>
spanned<T, 2> make_spanned(T (&arr)[N][M]) {
  return spanned<T, 2>((T*)arr, {N, M});
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
