#!/usr/bin/env bash

# This is the choreo generated bash script to compile cute code

if [ -z "${CUDA_HOME}" ]; then
  export CUDA_HOME=/usr/local/cuda-13
fi

if [ -z "${CUTE_HOME}" ]; then
  export CUTE_HOME=/home/fem/choreo/extern/cutlass
fi

if [ ! -n "${CUDA_HOME}" ] || [ ! -f ${CUDA_HOME}/bin/nvcc ]; then
  echo "failed to find the CUDA installation."
  echo "install cuda or set CUDA_HOME to cuda installation directory."
  exit 1
fi

if [ ! -n "${CUTE_HOME}" ] || [ ! -f ${CUTE_HOME}/include/cutlass/cutlass.h ]; then
  echo "failed to find the CUTE installation."
  echo "install cuda or set CUTE_HOME to cute installation directory."
  exit 1
fi

NVCC=${CUDA_HOME}/bin/nvcc
NVCC_LIB=${CUDA_LIB}/lib

rm -fr /tmp/1780994114377551139_138024178890560
mkdir -p /tmp/1780994114377551139_138024178890560

cat <<'EOF' > /tmp/1780994114377551139_138024178890560/choreo.h
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
  #define __CHOREO_TARGET_NATIVE_BF16_SUPPORT__
  #define __co_device__ __device__
  #define __co_host__ __host__
  #define __co_any__ __device__ __host__

#elif defined(__CHOREO_TARGET_AMDGPU__)
  #include <hip/hip_bfloat16.h>
  #include <hip/hip_fp16.h>
  #include <hip/hip_runtime.h>
  #define __CHOREO_TARGET_NATIVE_F16_SUPPORT__
  #define __CHOREO_TARGET_NATIVE_BF16_SUPPORT__
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

#elif defined(__CHOREO_TARGET_CC__)

  #define __co_device__
  #define __co_host__
  #define __co_any__

#else

  #define __co_device__
  #define __co_host__
  #define __co_any__

#endif // PRIVATE_TGT0, CUTE, CC

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
#elif defined(__CHOREO_TARGET_AMDGPU__)
  #define __CHOREO_BLOCK_SINGLE__                                              \
    threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0
  #define __CHOREO_GROUP_SINGLE__ threadIdx.x % 32 == 0
  #define __CHOREO_GROUP_ID__                                                  \
    (threadIdx.x + threadIdx.y * blockDim.x +                                  \
     threadIdx.z * blockDim.x * blockDim.y) /                                  \
        32
#elif defined(__CHOREO_TARGET_CUTE__)
  #define __CHOREO_BLOCK_SINGLE__                                              \
    threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0
  #define __CHOREO_GROUP_SINGLE__ threadIdx.x % 32 == 0
  #define __CHOREO_GROUPX4_SINGLE__ threadIdx.x % 128 == 0
  #define __CHOREO_GROUP_ID__                                                  \
    (threadIdx.x + threadIdx.y * blockDim.x +                                  \
     threadIdx.z * blockDim.x * blockDim.y) /                                  \
        32
#elif defined(__CHOREO_TARGET_CC__)
  #define __CHOREO_BLOCK_SINGLE__ true
  #define __CHOREO_GROUP_SINGLE__ true
#else
  #define __CHOREO_BLOCK_SINGLE__                                              \
    threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0
  #define __CHOREO_GROUP_SINGLE__ "invalid to use sublocal predicate"
#endif

#define __cok__ namespace choreo

namespace choreo {

namespace rtti {

template <int N>
struct mdspan {
  int data[N];
  __co_any__ int& operator[](int i) { return data[i]; }
  __co_any__ const int& operator[](int i) const { return data[i]; }
};

template <int N>
struct ituple {
  int data[N];
  __co_any__ int& operator[](int i) { return data[i]; }
  __co_any__ const int& operator[](int i) const { return data[i]; }
};

template <int N>
struct bounded_ituple {
  int data[N];
  int ub[N];
  __co_any__ int& operator[](int i) { return data[i]; }
  __co_any__ const int& operator[](int i) const { return data[i]; }
};

template <typename T, int N>
struct spanned {
  mdspan<N> span;
  mdspan<N> stride;
  T* data;
};

} // namespace rtti

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

template <typename T>
__co_any__ inline void fill(T* begin, T* end, const T& value) {
#if defined(__CUDA_ARCH__)
  for (size_t idx = 0; idx < static_cast<size_t>(end - begin); ++idx)
    begin[idx] = value;
#else
  std::fill(begin, end, value);
#endif
}

template <typename T>
__co_any__ inline void fill_n(T* begin, size_t n, const T& value) {
#if defined(__CUDA_ARCH__)
  for (size_t idx = 0; idx < n; ++idx) begin[idx] = value;
#else
  std::fill_n(begin, n, value);
#endif
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
__co_any__ inline size_t span_size(const mdspan<Rank>& s) {
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

} // namespace choreo

// Target-specific runtime (includes type overrides that must precede
// choreo_types.h)
#if __has_include("private_target0_runtime.h")
  #include "private_target0_runtime.h"
#endif

// Scalar type definitions, fallback classes, and conversion utilities.
#include "choreo_types.h"

// CuTe/CUDA-specific type operations: bitcast, FP8 arithmetic operators.
#if __has_include("choreo_types_cute.h")
  #include "choreo_types_cute.h"
#endif

namespace choreo {

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

// ---------------------------------------------------------------------------
// SampledVerifier: stride-based sampling verification for large matmul results.
//
// Treats the M*N result as a flat 1D array and samples elements at uniform
// stride intervals.  For each sampled element, computes a CPU reference dot
// product.  This gives O(num_samples * K) verification work instead of the
// full O(M * N * K), keeping verification tractable on CPU even at large
// problem sizes while covering positions spread across the full output.
// ---------------------------------------------------------------------------
struct SampledVerifierConfig {
  size_t num_samples = 512;
  float base_tol = 1.0f;
  float rel_tol = 0.01f;
  bool verbose = false;
};

// Pick a prime stride coprime with n so samples spread across both rows and
// columns.  Falls back to total/num_samples when that already satisfies the
// coprimality requirement.
inline size_t pick_coprime_stride(size_t total, size_t n, size_t num_samples) {
  size_t raw = std::max<size_t>(1, total / num_samples);
  auto gcd = [](size_t a, size_t b) {
    while (b) {
      size_t t = b;
      b = a % b;
      a = t;
    }
    return a;
  };
  if (gcd(raw, n) == 1) return raw;
  for (size_t s = raw + 1; s < total; ++s)
    if (gcd(s, n) == 1) return s;
  return raw;
}

// Row-col layout: lhs[M,K] (row-major) * rhs[K,N] (col-major) => res[M,N]
template <typename A, typename B, typename C>
__co_host__ inline void
verify_matmul_row_col_sampled(A& lhs, B& rhs, C& res,
                              const SampledVerifierConfig& cfg = {}) {
  size_t m = res.shape()[0];
  size_t n = res.shape()[1];
  size_t k = lhs.shape()[1];
  size_t total = m * n;
  size_t stride = pick_coprime_stride(total, n, cfg.num_samples);
  size_t checked = 0, failed = 0;
  for (size_t idx = 0; idx < total && checked < cfg.num_samples;
       idx += stride) {
    size_t i = idx / n, j = idx % n;
    float ref = 0.0f;
    for (size_t kk = 0; kk < k; ++kk)
      ref += to_f32(lhs[(int)i][(int)kk]) * to_f32(rhs[(int)kk][(int)j]);
    float got = to_f32(res[(int)i][(int)j]);
    float tol = cfg.base_tol + cfg.rel_tol * std::abs(ref);
    float diff = std::abs(got - ref);
    if (diff > tol) {
      if (cfg.verbose || failed < 5)
        std::cout << "mismatch at (" << i << ", " << j << ") gpu=" << got
                  << " ref=" << ref << " diff=" << diff << "\n";
      ++failed;
    }
    ++checked;
  }
  if (failed > 0) {
    std::cout << "FAILED: " << failed << "/" << checked << " samples\n";
    choreo_assert(false, "sampled verification failed");
  }
}

// Row-row layout: lhs[M,K] * rhs[N,K] (both row-major) => res[M,N]
template <typename A, typename B, typename C>
__co_host__ inline void
verify_matmul_row_row_sampled(A& lhs, B& rhs, C& res,
                              const SampledVerifierConfig& cfg = {}) {
  size_t m = res.shape()[0];
  size_t n = res.shape()[1];
  size_t k = lhs.shape()[1];
  size_t total = m * n;
  size_t stride = pick_coprime_stride(total, n, cfg.num_samples);
  size_t checked = 0, failed = 0;
  for (size_t idx = 0; idx < total && checked < cfg.num_samples;
       idx += stride) {
    size_t i = idx / n, j = idx % n;
    float ref = 0.0f;
    for (size_t kk = 0; kk < k; ++kk)
      ref += to_f32(lhs[(int)i][(int)kk]) * to_f32(rhs[(int)j][(int)kk]);
    float got = to_f32(res[(int)i][(int)j]);
    float tol = cfg.base_tol + cfg.rel_tol * std::abs(ref);
    float diff = std::abs(got - ref);
    if (diff > tol) {
      if (cfg.verbose || failed < 5)
        std::cout << "mismatch at (" << i << ", " << j << ") gpu=" << got
                  << " ref=" << ref << " diff=" << diff << "\n";
      ++failed;
    }
    ++checked;
  }
  if (failed > 0) {
    std::cout << "FAILED: " << failed << "/" << checked << " samples\n";
    choreo_assert(false, "sampled verification failed");
  }
}

// Sparse GEMM verification: dense_lhs[M,K] * rhs[N,K] => res[M,N]
// dense_lhs is the pre-sparsification dense matrix (row-major).
// rhs is row-major (N,K). Dot product: sum_k dense_lhs[i][k] * rhs[j][k].
template <typename A, typename B, typename C>
__co_host__ inline void
verify_spmm_sampled(A& dense_lhs, B& rhs, C& res,
                    const SampledVerifierConfig& cfg = {}) {
  verify_matmul_row_row_sampled(dense_lhs, rhs, res, cfg);
}

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
  const T* data() const { return ptr.get(); }

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
  __co_host__ void fill(U value) {
    fill_n(this->data(), this->element_count(), static_cast<T>(value));
  }

  // FP8-friendly fill: accepts float and converts for fp8 types.
  __co_host__ void fill_fp8(float value) {
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

// WGMMA uses u8 metadata (per-byte encoding, different from MMA's u32).
// For WGMMA sparse: f16/bf16 use K=32, fp8 uses K=64.
template <>
struct SparseMetaK<choreo::f16, choreo::u8> {
  static constexpr size_t value = 32;
};
template <>
struct SparseMetaK<choreo::bf16, choreo::u8> {
  static constexpr size_t value = 32;
};
template <>
struct SparseMetaK<choreo::f8_e4m3, choreo::u8> {
  static constexpr size_t value = 64;
};
template <>
struct SparseMetaK<choreo::f8_e5m2, choreo::u8> {
  static constexpr size_t value = 64;
};
#endif

// Convenience forwarding alias (non-breaking):
template <typename ValueT, typename MetaT>
using SparseHostPolicy =
    Sparse2to4HostPolicy<ValueT, MetaT, SparseMetaK<ValueT, MetaT>::value>;

// MMA sparse policy (uses u32 metadata).
template <typename ValueT, typename MetaT = choreo::u32>
using SparsePolicyMMA = SparseHostPolicy<ValueT, MetaT>;

// Deprecated alias for backward compatibility.
template <typename ValueT, typename MetaT = choreo::u32>
using SparsePolicy = SparsePolicyMMA<ValueT, MetaT>;

// Common fixed META_K aliases (for f16/bf16 sparse MMA variants).
template <typename ValueT, typename MetaT = choreo::u32>
using SparsePolicyK16 = Sparse2to4HostPolicy<ValueT, MetaT, 16>;

template <typename ValueT, typename MetaT = choreo::u32>
using SparsePolicyK32 = Sparse2to4HostPolicy<ValueT, MetaT, 32>;

// -----------------------------------------------------------------------------
// WGMMA-specific sparse policy (uses u8 metadata, per-byte encoding).
// -----------------------------------------------------------------------------

template <typename ValueT, size_t META_K>
struct Sparse2to4HostPolicyWGMMA {
  static_assert(META_K == 32 || META_K == 64, "WGMMA META_K must be 32 or 64.");

  __co_host__ static inline void
  init_structured_sparse_A(spanned_data<ValueT, 2>& dense, std::mt19937& gen) {
    const size_t M = dense.shape()[0];
    const size_t K = dense.shape()[1];
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::uniform_int_distribution<int> pick(0, 3);
    dense.fill(ValueT(0));
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
        if (is_zero(t0)) t0 = from_f32<ValueT>(1.0f);
        if (is_zero(t1)) t1 = from_f32<ValueT>(-1.0f);
        dense.data()[base + idx0] = t0;
        dense.data()[base + idx1] = t1;
      }
    }
  }

  __co_host__ static inline void encode(spanned_data<ValueT, 2>& dense,
                                        spanned_data<ValueT, 2>& packed,
                                        spanned_data<choreo::u8, 2>& meta) {
    const size_t M = dense.shape()[0];
    const size_t K = dense.shape()[1];
    for (size_t r = 0; r < M; ++r) {
      for (size_t c_group = 0; c_group < K / 4; ++c_group) {
        size_t base = r * K + c_group * 4;
        size_t in_base = r * (K / 2) + c_group * 2;
        int idxs[2] = {-1, -1};
        int nz = 0;
        for (int i = 0; i < 4; ++i) {
          if (to_f32(dense.data()[base + i]) != 0.0f) {
            if (nz < 2) idxs[nz] = i;
            ++nz;
          }
        }
        if (nz != 2) {
          idxs[0] = 0;
          idxs[1] = 1;
        }
        if (idxs[0] > idxs[1]) std::swap(idxs[0], idxs[1]);
        ValueT v0 = dense.data()[base + idxs[0]];
        ValueT v1 = dense.data()[base + idxs[1]];
        packed.data()[in_base + 0] = v0;
        packed.data()[in_base + 1] = v1;
        uint8_t nibble = static_cast<uint8_t>(idxs[0] | (idxs[1] << 2));
        size_t byte_col = c_group / 2;
        if ((c_group & 1) == 0) {
          meta[r][byte_col] = nibble;
        } else {
          meta[r][byte_col] |= static_cast<choreo::u8>(nibble << 4);
        }
      }
    }
  }

  __co_host__ static inline void
  prepack(spanned_data<choreo::u8, 2>& meta_u8,
          spanned_data<choreo::u32, 2>& meta_u32) {
    const size_t M = meta_u8.shape()[0];
    const size_t K_meta = meta_u8.shape()[1];
    const size_t K_meta_u32 = meta_u32.shape()[1];
    const size_t k_fragments = K_meta / 4;

    choreo_assert(meta_u32.shape()[0] == M,
                  "Sparse prepack requires matching M dimensions");
    choreo_assert((M % 16) == 0,
                  "Sparse prepack requires M dimension divisible by 16");
    choreo_assert((K_meta % 4) == 0,
                  "Sparse prepack requires u8 metadata K dimension divisible "
                  "by 4");
    choreo_assert(K_meta_u32 == k_fragments,
                  "Sparse prepack expects one u32 metadata column per 32-wide "
                  "K fragment");

    for (size_t block_m = 0; block_m < M; block_m += 16) {
      for (size_t row = 0; row < 8; ++row) {
        const size_t row_lo = block_m + row;
        const size_t row_hi = block_m + row + 8;
        for (size_t k_frag = 0; k_frag < k_fragments; ++k_frag) {
          const size_t byte_col_base = k_frag * 4;
          uint16_t lo16 =
              uint16_t(meta_u8.data()[row_lo * K_meta + byte_col_base + 0]) |
              (uint16_t(meta_u8.data()[row_lo * K_meta + byte_col_base + 1])
               << 8);
          uint16_t hi16 =
              uint16_t(meta_u8.data()[row_lo * K_meta + byte_col_base + 2]) |
              (uint16_t(meta_u8.data()[row_lo * K_meta + byte_col_base + 3])
               << 8);
          uint16_t lo16_pair =
              uint16_t(meta_u8.data()[row_hi * K_meta + byte_col_base + 0]) |
              (uint16_t(meta_u8.data()[row_hi * K_meta + byte_col_base + 1])
               << 8);
          uint16_t hi16_pair =
              uint16_t(meta_u8.data()[row_hi * K_meta + byte_col_base + 2]) |
              (uint16_t(meta_u8.data()[row_hi * K_meta + byte_col_base + 3])
               << 8);

          meta_u32.data()[(block_m + 2 * row + 0) * K_meta_u32 + k_frag] =
              uint32_t(lo16) | (uint32_t(lo16_pair) << 16);
          meta_u32.data()[(block_m + 2 * row + 1) * K_meta_u32 + k_frag] =
              uint32_t(hi16) | (uint32_t(hi16_pair) << 16);
        }
      }
    }
  }

  // prepack_v2: coalesced-access metadata layout.
  // Rearranges the v1-prepacked u32 metadata so that, for each 16-row block
  // and k-fragment, the 16 u32 values needed by the 16 active warp lanes are
  // stored contiguously. Layout: [M/16, K_frags, 16].
  // Device-side: thread reads meta_v2[(m16 * K_frags + kf) * 16 + pos]
  //   where all 16 active lanes differ only in pos -> single cache-line txn.
  __co_host__ static inline void
  prepack_v2(spanned_data<choreo::u8, 2>& meta_u8,
             spanned_data<choreo::u32, 2>& meta_u32) {
    const size_t M = meta_u8.shape()[0];
    const size_t K_meta = meta_u8.shape()[1];
    const size_t K_meta_u32 = meta_u32.shape()[1];
    const size_t k_fragments = K_meta / 4;

    choreo_assert(meta_u32.shape()[0] == M,
                  "Sparse prepack_v2 requires matching M dimensions");
    choreo_assert((M % 16) == 0,
                  "Sparse prepack_v2 requires M divisible by 16");
    choreo_assert((K_meta % 4) == 0,
                  "Sparse prepack_v2 requires u8 K dim divisible by 4");
    choreo_assert(K_meta_u32 == k_fragments,
                  "Sparse prepack_v2 expects K_meta_u32 == k_fragments");

    auto temp = choreo::make_spandata<choreo::u32>(M, k_fragments);
    prepack(meta_u8, temp);

    for (size_t block_m = 0; block_m < M; block_m += 16) {
      const size_t m16 = block_m / 16;
      for (size_t rib = 0; rib < 16; ++rib) {
        for (size_t kf = 0; kf < k_fragments; ++kf) {
          size_t v1_idx = (block_m + rib) * k_fragments + kf;
          size_t v2_idx = (m16 * k_fragments + kf) * 16 + rib;
          meta_u32.data()[v2_idx] = temp.data()[v1_idx];
        }
      }
    }
  }
};

// prepack_v2 reorder for fp8 u32 metadata (already encoded as u32).
// Rearranges [M, K_meta_cols] u32 to coalesced layout [M/16, K64_frags, 32]
// where 32 values per (m_block_16, k64_frag) are stored in the thread access
// order of wgmma.sp fp8, enabling a single 128-byte cache-line transaction.
__co_host__ inline void
prepack_v2_fp8_reorder(spanned_data<choreo::u32, 2>& meta_in,
                       spanned_data<choreo::u32, 2>& meta_out) {
  const size_t M = meta_in.shape()[0];
  const size_t K_meta_cols = meta_in.shape()[1];
  const size_t num_k64_frags = K_meta_cols / 2;

  choreo_assert(meta_out.shape()[0] == M && meta_out.shape()[1] == K_meta_cols,
                "prepack_v2_fp8_reorder: output shape must match input");
  choreo_assert((M % 16) == 0,
                "prepack_v2_fp8_reorder: M must be divisible by 16");

  for (size_t m16 = 0; m16 < M / 16; ++m16) {
    for (size_t k64 = 0; k64 < num_k64_frags; ++k64) {
      for (size_t rib = 0; rib < 16; ++rib) {
        for (size_t col = 0; col < 2; ++col) {
          size_t global_row = m16 * 16 + rib;
          size_t global_col = k64 * 2 + col;
          int low_row = rib & 7;
          int high_bit = (rib >> 3) & 1;
          int tid = (low_row << 2) | (col << 1) | high_bit;
          size_t v2_idx = (m16 * num_k64_frags + k64) * 32 + tid;
          meta_out.data()[v2_idx] =
              meta_in.data()[global_row * K_meta_cols + global_col];
        }
      }
    }
  }
}

// prepack_v2 reorder for f16/bf16 u32 metadata (16-bit value types).
// Rearranges [M, K_frags] u32 to coalesced layout [M/16, K_frags, 16].
__co_host__ inline void
prepack_v2_16bit_reorder(spanned_data<choreo::u32, 2>& meta_in,
                         spanned_data<choreo::u32, 2>& meta_out) {
  const size_t M = meta_in.shape()[0];
  const size_t K_frags = meta_in.shape()[1];

  choreo_assert(meta_out.shape()[0] == M && meta_out.shape()[1] == K_frags,
                "prepack_v2_16bit_reorder: output shape must match input");
  choreo_assert((M % 16) == 0,
                "prepack_v2_16bit_reorder: M must be divisible by 16");

  for (size_t m16 = 0; m16 < M / 16; ++m16) {
    for (size_t rib = 0; rib < 16; ++rib) {
      for (size_t kf = 0; kf < K_frags; ++kf) {
        size_t v1_idx = (m16 * 16 + rib) * K_frags + kf;
        size_t v2_idx = (m16 * K_frags + kf) * 16 + rib;
        meta_out.data()[v2_idx] = meta_in.data()[v1_idx];
      }
    }
  }
}

// WGMMA convenience aliases.
template <typename ValueT>
using SparsePolicyWGMMAK32 = Sparse2to4HostPolicyWGMMA<ValueT, 32>;

template <typename ValueT>
using SparsePolicyWGMMAK64 = Sparse2to4HostPolicyWGMMA<ValueT, 64>;

// WGMMA sparse policy that infers META_K from dtype (uses u8 metadata).
template <typename ValueT>
using SparsePolicyWGMMA =
    Sparse2to4HostPolicyWGMMA<ValueT, SparseMetaK<ValueT, choreo::u8>::value>;

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
// WGMMA u8 metadata regressions.
static_assert(SparseMetaK<choreo::f16, choreo::u8>::value == 32,
              "Regression: SparseMetaK<f16,u8> changed");
static_assert(SparseMetaK<choreo::bf16, choreo::u8>::value == 32,
              "Regression: SparseMetaK<bf16,u8> changed");
static_assert(SparseMetaK<choreo::f8_e4m3, choreo::u8>::value == 64,
              "Regression: SparseMetaK<f8_e4m3,u8> changed");
static_assert(SparseMetaK<choreo::f8_e5m2, choreo::u8>::value == 64,
              "Regression: SparseMetaK<f8_e5m2,u8> changed");
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
#elif defined(__CHOREO_TARGET_AMDGPU__)
    auto err = static_cast<hipError_t>(p);
    fprintf(stderr, "HIP failure: %d (%s)\n", (int)err, hipGetErrorString(err));
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
#elif defined(__CHOREO_TARGET_AMDGPU__)
  hipError_t err = hipGetLastError();
  if (err != hipSuccess) {
    fprintf(stderr, "HIP error after kernel: %s\n", hipGetErrorString(err));
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

namespace croq = choreo;

inline void __co_any__ croq_assert(bool p, const char* msg,
                                   const char* file = __FILE__,
                                   int line = __LINE__) {
  choreo::choreo_assert(p, msg, file, line);
}

#endif // __CHOREO_H__

EOF

cat <<'EOF' > /tmp/1780994114377551139_138024178890560/choreo_types.h
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

EOF

cat <<'EOF' > /tmp/1780994114377551139_138024178890560/choreo_types_cute.h
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

EOF

cat <<'EOF' > /tmp/1780994114377551139_138024178890560/choreo_cute.h
#ifndef __CHROEO_CUTE_HEADER_LIBRARY_HPP__
#define __CHROEO_CUTE_HEADER_LIBRARY_HPP__

// Note: No manual inclusion. (included by "choreo.h")

#ifdef __CHOREO_TARGET_CUTE__

namespace choreo {

using stream_t = cudaStream_t;

  #if defined(__CHOREO_ENABLE_CUDA_RUNTIME_ENV_CHECK__) &&                     \
      defined(__CHOREO_REQUIRED_GPU_DEVICE_SM__) &&                            \
      !defined(__CHOREO_CUDA_RUNTIME_ENV_CHECKER_DEFINED__)
    #define __CHOREO_CUDA_RUNTIME_ENV_CHECKER_DEFINED__

static inline void __choreo_check_cuda_environment__() {
  static bool already_checked = false;
  if (already_checked) return;
  already_checked = true;

  auto decode_cuda_version = [](int v, int& major, int& minor, int& patch) {
    major = v / 1000;
    minor = (v % 1000) / 10;
    patch = v % 10;
  };

  int runtime_ver = 0;
  cudaError_t err = cudaRuntimeGetVersion(&runtime_ver);
  if (err != cudaSuccess) {
    std::fprintf(stderr, "[choreo] CUDA runtime not available: %s\n",
                 cudaGetErrorString(err));
    std::exit(EXIT_FAILURE);
  }

  int driver_ver = 0;
  err = cudaDriverGetVersion(&driver_ver);
  if (err != cudaSuccess) {
    std::fprintf(stderr, "[choreo] CUDA driver not available: %s\n",
                 cudaGetErrorString(err));
    std::exit(EXIT_FAILURE);
  }

  int rMaj, rMin, rPat;
  int dMaj, dMin, dPat;
  decode_cuda_version(runtime_ver, rMaj, rMin, rPat);
  decode_cuda_version(driver_ver, dMaj, dMin, dPat);

  int reqMaj, reqMin, reqPat;
  decode_cuda_version(CUDART_VERSION, reqMaj, reqMin, reqPat);

  if (runtime_ver < CUDART_VERSION) {
    std::fprintf(stderr,
                 "[choreo] CUDA runtime too old:\n"
                 "  found runtime %d.%d.%d (encoded=%d)\n"
                 "  required      %d.%d.%d (encoded=%d)\n",
                 rMaj, rMin, rPat, runtime_ver, reqMaj, reqMin, reqPat,
                 CUDART_VERSION);
    std::exit(EXIT_FAILURE);
  }

  if (driver_ver < runtime_ver) {
    std::fprintf(
        stderr,
        "[choreo] Warning: CUDA driver (%d.%d.%d, encoded=%d) is older "
        "than the CUDA runtime (%d.%d.%d, encoded=%d). This may cause "
        "issues.\n",
        dMaj, dMin, dPat, driver_ver, rMaj, rMin, rPat, runtime_ver);
  }

  int device_count = 0;
  err = cudaGetDeviceCount(&device_count);
  if (err != cudaSuccess || device_count == 0) {
    std::fprintf(stderr, "[choreo] No CUDA-capable devices found.\n");
    std::exit(EXIT_FAILURE);
  }

  int device_id = 0;
  cudaDeviceProp prop{};
  err = cudaGetDeviceProperties(&prop, device_id);
  if (err != cudaSuccess) {
    std::fprintf(stderr, "[choreo] cudaGetDeviceProperties failed: %s\n",
                 cudaGetErrorString(err));
    std::exit(EXIT_FAILURE);
  }

  int sm = prop.major * 10 + prop.minor;
  if (sm < __CHOREO_REQUIRED_GPU_DEVICE_SM__) {
    std::fprintf(stderr,
                 "[choreo] Compute capability too low on device %d (%s):\n"
                 "  found SM %d.%d (sm_%d)\n"
                 "  required SM >= %d (sm_%d)\n",
                 device_id, prop.name, prop.major, prop.minor, sm,
                 __CHOREO_REQUIRED_GPU_DEVICE_SM__,
                 __CHOREO_REQUIRED_GPU_DEVICE_SM__);
    std::exit(EXIT_FAILURE);
  }
}

  #endif

  #ifndef CHOREO_PTX_BARRIER_MAX_SPINS
    #define CHOREO_PTX_BARRIER_MAX_SPINS (1u << 24)
  #endif

__device__ __forceinline__ uint32_t tma_to_shared_u32(const void* p) {
  #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
  #else
  (void)p;
  return 0;
  #endif
}

__device__ __forceinline__ void tma_mbarrier_init(uint64_t* bar,
                                                  uint32_t thread_count) {
  #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  uint32_t bar_ptr = tma_to_shared_u32(bar);
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n"
               :
               : "r"(bar_ptr), "r"(thread_count));
  #else
  (void)bar;
  (void)thread_count;
  #endif
}

__device__ __forceinline__ void tma_mbarrier_expect_tx(uint64_t* bar,
                                                       uint32_t bytes) {
  #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  uint32_t bar_ptr = tma_to_shared_u32(bar);
  asm volatile(
      "mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;\n"
      :
      : "r"(bar_ptr), "r"(bytes));
  #else
  (void)bar;
  (void)bytes;
  #endif
}

__device__ __forceinline__ void
tma_mbarrier_expect_tx_noarrive(uint64_t* bar, uint32_t bytes) {
  #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  uint32_t bar_ptr = tma_to_shared_u32(bar);
  asm volatile("mbarrier.expect_tx.relaxed.cta.shared::cta.b64 [%0], %1;\n"
               :
               : "r"(bar_ptr), "r"(bytes));
  #else
  (void)bar;
  (void)bytes;
  #endif
}

__device__ __forceinline__ void
tma_load_2d_shared_cta_global_mbarrier(void* dst, const void* tma_map,
                                       uint64_t* bar, int32_t coord0,
                                       int32_t coord1) {
  #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  uint64_t tma_ptr = reinterpret_cast<uint64_t>(tma_map);
  uint32_t bar_ptr = tma_to_shared_u32(bar);
  uint32_t dst_ptr = tma_to_shared_u32(dst);
  asm volatile("cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::"
               "complete_tx::bytes "
               "[%0], [%1, {%3, %4}], [%2];"
               :
               : "r"(dst_ptr), "l"(tma_ptr), "r"(bar_ptr), "r"(coord0),
                 "r"(coord1)
               : "memory");
  #else
  (void)dst;
  (void)tma_map;
  (void)bar;
  (void)coord0;
  (void)coord1;
  #endif
}

__device__ __forceinline__ void
tma_load_2d_shared_cluster_global_mbarrier(void* dst, const void* tma_map,
                                           uint64_t* bar, int32_t coord0,
                                           int32_t coord1) {
  #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  uint64_t tma_ptr = reinterpret_cast<uint64_t>(tma_map);
  uint32_t bar_ptr = tma_to_shared_u32(bar);
  uint32_t dst_ptr = tma_to_shared_u32(dst);
  asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::"
               "complete_tx::bytes "
               "[%0], [%1, {%3, %4}], [%2];"
               :
               : "r"(dst_ptr), "l"(tma_ptr), "r"(bar_ptr), "r"(coord0),
                 "r"(coord1)
               : "memory");
  #else
  (void)dst;
  (void)tma_map;
  (void)bar;
  (void)coord0;
  (void)coord1;
  #endif
}

__device__ __forceinline__ void
tma_load_2d_shared_cluster_global_mbarrier_multicast(
    void* dst, const void* tma_map, uint64_t* bar, int32_t coord0,
    int32_t coord1, uint16_t multicast_mask) {
  #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  uint64_t tma_ptr = reinterpret_cast<uint64_t>(tma_map);
  uint32_t bar_ptr = tma_to_shared_u32(bar);
  uint32_t dst_ptr = tma_to_shared_u32(dst);
  asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.tile."
               "mbarrier::complete_tx::bytes.multicast::cluster"
               " [%0], [%1, {%3, %4}], [%2], %5;"
               :
               : "r"(dst_ptr), "l"(tma_ptr), "r"(bar_ptr), "r"(coord0),
                 "r"(coord1), "h"(multicast_mask)
               : "memory");
  #else
  (void)dst;
  (void)tma_map;
  (void)bar;
  (void)coord0;
  (void)coord1;
  (void)multicast_mask;
  #endif
}

__device__ __forceinline__ uint32_t tma_cluster_rank() {
  #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  uint32_t rank;
  asm volatile("mov.u32 %0, %cluster_ctarank;\n" : "=r"(rank));
  return rank;
  #else
  return 0;
  #endif
}

__device__ __forceinline__ uint32_t tma_cluster_dim() {
  #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  uint32_t dim;
  asm volatile("mov.u32 %0, %cluster_nctarank;\n" : "=r"(dim));
  return dim;
  #else
  return 1;
  #endif
}

__device__ __forceinline__ void tma_cluster_sync() {
  #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  asm volatile("barrier.cluster.arrive;\n" ::: "memory");
  asm volatile("barrier.cluster.wait;\n" ::: "memory");
  #endif
}

__device__ __forceinline__ void
tma_mbarrier_arrive_cluster(uint64_t* bar, uint32_t cta_id,
                            uint32_t count = 1) {
  #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  uint32_t smem_addr = tma_to_shared_u32(bar);
  asm volatile("{\n\t"
               ".reg .b32 remAddr32;\n\t"
               "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
               "mbarrier.arrive.shared::cluster.b64 _, [remAddr32], %2;\n\t"
               "}"
               :
               : "r"(smem_addr), "r"(cta_id), "r"(count));
  #else
  (void)bar;
  (void)cta_id;
  (void)count;
  #endif
}

__device__ __forceinline__ void
tma_mbarrier_wait_parity_cluster(uint64_t* bar, int phase_bit) {
  #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  uint32_t bar_ptr = tma_to_shared_u32(bar);
  uint32_t spins = 0;
  while (true) {
    uint32_t ready = 0;
    asm volatile(
        "{\n"
        ".reg .pred P1;\n"
        "mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 P1, [%1], "
        "%2;\n"
        "selp.b32 %0, 1, 0, P1;\n"
        "}\n"
        : "=r"(ready)
        : "r"(bar_ptr), "r"(phase_bit)
        : "memory");
    if (ready) return;
    if (++spins >= CHOREO_PTX_BARRIER_MAX_SPINS) {
      printf("WARN: cluster mbarrier wait exceeded max spins\n");
      return;
    }
  }
  #else
  (void)bar;
  (void)phase_bit;
  #endif
}

__device__ __forceinline__ void tma_mbarrier_arrive(uint64_t* bar,
                                                    uint32_t count = 1) {
  #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  uint32_t bar_ptr = tma_to_shared_u32(bar);
  asm volatile("mbarrier.arrive.release.cta.shared::cta.b64 _, [%0], %1;\n"
               :
               : "r"(bar_ptr), "r"(count)
               : "memory");
  #else
  (void)bar;
  (void)count;
  #endif
}

__device__ __forceinline__ void tma_mbarrier_wait_parity(uint64_t* bar,
                                                         int phase_bit) {
  #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  uint32_t bar_ptr = tma_to_shared_u32(bar);
  uint32_t spins = 0;
  while (true) {
    uint32_t ready = 0;
    asm volatile(
        "{\n"
        ".reg .pred P1;\n"
        "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 P1, [%1], %2;\n"
        "selp.b32 %0, 1, 0, P1;\n"
        "}\n"
        : "=r"(ready)
        : "r"(bar_ptr), "r"(phase_bit)
        : "memory");

    if (ready) return;

    ++spins;
    if (spins >= CHOREO_PTX_BARRIER_MAX_SPINS) {
      printf("[choreo-rt] PTX mbarrier wait timeout: bar=%u, phase=%d, "
             "spins=%u\\n",
             bar_ptr, phase_bit, spins);
      __trap();
    }

    if ((spins & 0x3FFu) == 0u) __nanosleep(64);
  }
  #else
  (void)bar;
  (void)phase_bit;
  #endif
}

// SM90+ (Hopper+) - TMA barrier and token
struct TMAAtom {
  cuda::barrier<cuda::thread_scope_block>* bar;
  cuda::barrier<cuda::thread_scope_block>::arrival_token tok;
  uint64_t* ptx_bar = nullptr;
  int ptx_phase = 0;
  bool use_ptx_mbarrier = false;
  //  TMAAtom(cuda::barrier<cuda::thread_scope_block> *b): bar(b) {}
  __device__ auto& barrier() { return *bar; }
  __device__ auto& token() { return tok; }
  __device__ uint64_t* ptx_barrier() { return ptx_bar; }
  __device__ int ptx_phase_bit() const { return ptx_phase; }
  __device__ void toggle_ptx_phase() { ptx_phase ^= 1; }
  __device__ bool IsPTXMBarrier() const { return use_ptx_mbarrier; }
  __device__ void EnablePTXMBarrier(uint64_t* b) {
    ptx_bar = b;
    ptx_phase = 0;
    use_ptx_mbarrier = true;
  }
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
  // note: return the remaining count N to feed cp_async_wait<N>
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
  void* d = nullptr;  // data: future's user must guarantee it is valid
  void* md = nullptr; // metadata: optional structured sparsity metadata

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
    ring = r + __CHOREO_GROUP_ID__;
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

  __device__ future(const char* n, unsigned l, unsigned c, void* data = nullptr,
                    void* mdata = nullptr)
      : d(data), md(mdata ? mdata : data), s(ST_NONE), name(n), line(l),
        column(c) {}
  #else
  __device__ future(const char* n, unsigned l, unsigned c, void* data = nullptr,
                    void* mdata = nullptr)
      : d(data), md(mdata ? mdata : data) {
    (void)n;
    (void)l;
    (void)c;
  }

  __device__ future(void* data = nullptr, void* mdata = nullptr)
      : d(data), md(mdata ? mdata : data) {}
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
    s = ST_INITED;
  #endif // __CHOREO_DMA_DIAGNOSIS__

    atom = a;
  }

  // when sync, no wait is required. simply change the status
  __device__ void set_nowait() {
  #ifdef __CHOREO_DMA_DIAGNOSIS__
    if (s != ST_INITED && s != ST_WAITED) {
      if (s == ST_NONE)
        printf("[choreo-rt] WARN: future (defined at line %u:%u) maybe in none "
               "state when setting nowait.\n",
               line, column);
      else if (s == ST_TRIGGERED)
        printf("[choreo-rt] WARN: future (defined at line %u:%u) maybe in "
               "triggered state when setting nowait.\n",
               line, column);
      else
        printf("[choreo-rt] WARN: future (defined at line %u:%u) maybe in an "
               "invalid state when setting nowait.\n",
               line, column);
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
      auto* tma_atom = ((TMAAtom*)atom);
      if (tma_atom->IsPTXMBarrier()) {
    #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
        tma_mbarrier_wait_parity(tma_atom->ptx_barrier(),
                                 tma_atom->ptx_phase_bit());
    #else
        __co_abort__();
    #endif
      } else {
        auto& barrier = tma_atom->barrier();
        auto& token = tma_atom->token();
        barrier.wait(std::move(token));
      }
      return;
    }
    // TODO: make shared memory be allocated by host
    __shared__ int discard_count;
    discard_count = 0;
    // cautious: must be warp based
    if (__CHOREO_GROUP_SINGLE__) {
      assert(ring && "ring is invalid.");
      discard_count = ring->discard(this);
    }
    // sync it to warp threads
    discard_count = __shfl_sync(0xffffffff, discard_count, 0);
    switch (discard_count) {
    case -1: break;
    case 0: cute::cp_async_wait<0>(); break;
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
    __syncwarp();

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
    s = ST_TRIGGERED;
  #endif // __CHOREO_DMA_DIAGNOSIS__

  #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    if (is_tma) return; // TMA uses event barriers, not cp.async rings
    // cautious: must be warp based
    if (__CHOREO_GROUP_SINGLE__) {
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
      printf("[choreo-rt] WARN: maybe wait the future (defined at line %u:%u) "
             "before trigger it.\n",
             line, column);
    } else
      assert(s == ST_NONE); // waiting on not triggered future is acceptable
  #endif                    // __CHOREO_DMA_DIAGNOSIS__
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

  __device__ void* mdata() {
  #ifdef __CHOREO_DMA_DIAGNOSIS__
    if (!md) {
      printf("[choreo-rt] internal error: future (defined at line %u:%u) is "
             "not associated with a metadata.\n",
             line, column);
      __co_abort__();
    }
    if (s == ST_TRIGGERED) {
      printf("[choreo-rt] Error is detected: future (defined at line %u:%u) is "
             "not waited before using.\n",
             line, column);
      __co_abort__();
    }
  #endif // __CHOREO_DMA_DIAGNOSIS__
    return md ? md : d;
  }

  __device__ void destroy() {}

  __device__ ~future() {
  #ifdef __CHOREO_DMA_DIAGNOSIS__
    if (s == ST_TRIGGERED)
      printf("[choreo-rt] WARN: the future (defined at line %u:%u) may never "
             "have been waited.\n",
             line, column);
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
  #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
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
  #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800

    #ifdef __CHOREO_DEBUG_FUTURE_RING__
  printf("discarding feature: %d, [%d, %d)\n", f->id, tail, head);
    #endif

  uint8_t p = tail;
  while (p != head) {
    if (ring[p] == f->id) {
      int size = (head - p - 1 + N) % N;
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

    #ifdef __CHOREO_DMA_DIAGNOSIS__
  printf("[choreo-rt] Internal error: future %d (defined at line %u:%u) "
         "is not committed.\n",
         f->id, f->line, f->column);
    #else
  printf("[choreo-rt] Internal error: future %d is not committed.\n", f->id);
    #endif

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

template <size_t Bits, class Ptr>
CUTE_HOST_DEVICE std::uintptr_t aligned_up_ptr(Ptr p) {
  constexpr std::uintptr_t Align = Bits / 8;
  auto ptr = reinterpret_cast<std::uintptr_t>(p);
  return (ptr + Align - 1) & ~(Align - 1);
}

// ------------------- universal copy (any rank, C++17) -------------------
template <class ATOM, class Src, class Dst>
CUTE_HOST_DEVICE void naive_copy(ATOM& atom, const Src& src, Dst& dst) {
  cute::copy(atom, src, dst);
}

template <class Src, class Dst>
CUTE_HOST_DEVICE void naive_copy(const Src& src, Dst& dst) {
  using SrcVal = typename Src::value_type;
  using DstVal = typename Dst::value_type;
  CUTE_UNROLL
  for (int i = 0; i < cute::size(src); ++i) {
    if constexpr (std::is_same<SrcVal, uint8_t>::value &&
                  cutlass::sizeof_bits<DstVal>::value < 8) {
      dst(i) = DstVal::bitcast(src(i));
    } else if constexpr (std::is_same<DstVal, uint8_t>::value &&
                         cutlass::sizeof_bits<SrcVal>::value < 8) {
      SrcVal tmp = src(i);
      dst(i) = tmp.raw();
    } else {
      dst(i) = src(i);
    }
  }
}

// COPY ATOM:
// SM80_CP_ASYNC_CACHEGLOBAL_ZFILL used for cp.async
// AutoVectorizingCopyWithAssumedAlignment for auto-vectorized copy with assumed
// alignment (e.g. 128-bit for global memory) UniversalCopy for general-purpose
// copy with any shape/stride

// Trait: detect AutoVectorizingCopy atoms wider than element size.
// These atoms cause misaligned stores in CuTe's copy_if (per-element dispatch
// with a wide atom). cp_async and narrow atoms are safe with copy_if.
template <class>
struct choreo_is_wide_autovec_ : std::false_type {};
template <int N, class T>
struct choreo_is_wide_autovec_<
    cute::Copy_Atom<cute::AutoVectorizingCopyWithAssumedAlignment<N>, T>>
    : std::bool_constant<(N > int(sizeof(T) * 8))> {};

// ------------------- threads-collective tiled copy -------------------
template <class ATOM, int ThrRows, int ThrCols, int ValRows, int ValCols,
          bool Swizzle, bool Pred, bool ZFill = false, class SrcTensor,
          class DstTensor, class PredFn>
__device__ static inline void tiled_copy(const SrcTensor& src, DstTensor& dst,
                                         PredFn pred) {
  using namespace cute;
  auto tc =
      make_tiled_copy(ATOM{},
                      make_layout(make_shape(Int<ThrRows>{}, Int<ThrCols>{}),
                                  make_stride(Int<ThrCols>{}, Int<1>{})),
                      make_layout(make_shape(Int<ValRows>{}, Int<ValCols>{})));
  constexpr int NumThreads = ThrRows * ThrCols;
  auto thr_copy = tc.get_thread_slice(threadIdx.x % NumThreads);
  auto src_thr = [&]() {
    if constexpr (Swizzle && is_smem<remove_cvref_t<SrcTensor>>::value) {
      auto src_pi =
          as_position_independent_swizzle_tensor(const_cast<SrcTensor&>(src));
      return thr_copy.partition_S(src_pi);
    } else {
      return thr_copy.partition_S(src);
    }
  }();
  auto dst_thr = [&]() {
    if constexpr (Swizzle && is_smem<remove_cvref_t<DstTensor>>::value) {
      auto dst_pi = as_position_independent_swizzle_tensor(dst);
      return thr_copy.partition_D(dst_pi);
    } else {
      return thr_copy.partition_D(dst);
    }
  }();
  if constexpr (Pred) {
    auto coord_thr = thr_copy.partition_S(make_identity_tensor(shape(src)));
    auto pred_thr = make_tensor<bool>(shape(src_thr));
    CUTE_UNROLL
    for (int i = 0; i < size(pred_thr); ++i) {
      pred_thr(i) = pred(coord_thr(i));
    }
    if constexpr (choreo_is_wide_autovec_<ATOM>::value) {
      // CuTe's copy_if applies the wide atom per-element, causing misaligned
      // stores when atom width > element width. Use atom-group-aware copy:
      // vectorize full atom groups, scalar-copy partial boundary groups.
      constexpr int kAtomV = size<0>(src_thr);
      if constexpr (rank(src_thr) == 3) {
        CUTE_UNROLL
        for (int k = 0; k < size<2>(src_thr); ++k) {
          CUTE_UNROLL
          for (int m = 0; m < size<1>(src_thr); ++m) {
            bool all_valid = true;
            CUTE_UNROLL
            for (int v = 0; v < kAtomV; ++v) {
              if (!pred_thr(v, m, k)) {
                all_valid = false;
                break;
              }
            }
            if (all_valid) {
              copy(ATOM{}, src_thr(_, m, k), dst_thr(_, m, k));
            } else {
              CUTE_UNROLL
              for (int v = 0; v < kAtomV; ++v) {
                if (pred_thr(v, m, k)) { dst_thr(v, m, k) = src_thr(v, m, k); }
              }
            }
          }
        }
      } else {
        CUTE_UNROLL
        for (int m = 0; m < size<1>(src_thr); ++m) {
          bool all_valid = true;
          CUTE_UNROLL
          for (int v = 0; v < kAtomV; ++v) {
            if (!pred_thr(v, m)) {
              all_valid = false;
              break;
            }
          }
          if (all_valid) {
            copy(ATOM{}, src_thr(_, m), dst_thr(_, m));
          } else {
            CUTE_UNROLL
            for (int v = 0; v < kAtomV; ++v) {
              if (pred_thr(v, m)) { dst_thr(v, m) = src_thr(v, m); }
            }
          }
        }
      }
    } else {
      copy_if(tc, pred_thr, src_thr, dst_thr);
    }
    if constexpr (ZFill) {
      using DstValT = typename DstTensor::value_type;
      CUTE_UNROLL
      for (int i = 0; i < size(pred_thr); ++i) {
        if (!pred_thr(i)) { dst_thr(i) = DstValT{}; }
      }
    }
  } else {
    copy(tc, src_thr, dst_thr);
  }
}

// ------------------- threads-collective fill -------------------
template <class Tensor, class T>
__device__ static inline void cooperative_fill(Tensor& dst, T val) {
  auto n = cute::size(dst);
  for (int i = threadIdx.x; i < n; i += blockDim.x) { dst(i) = val; }
  __syncthreads();
}

// TODO: move to choreo_mma_wrapper.h
// ------------------- inline mma PTX -------------------

template <class MMA>
struct MMA_Policy {
  static constexpr bool supported = false;
};

// Accumulation type casting helper
template <class T, class F>
struct AccumTCast {
  static constexpr bool supported = false;
};

template <>
struct AccumTCast<f16, f32> {
  static constexpr bool supported = true;
  __device__ static inline f16 cast(f32 val) { return f32_to_f16(val); }
};

template <>
struct AccumTCast<f32, f16> {
  static constexpr bool supported = true;
  __device__ static inline f32 cast(f16 val) { return f16_to_f32(val); }
};

template <class T, class F>
__device__ static inline T cast_if(F val) {
  if constexpr (AccumTCast<F, T>::supported) {
    return AccumTCast<F, T>::cast(val);
  } else {
    static_assert(std::is_same<T, F>::value,
                  "unsupported accumulation type casting");
    return val;
  }
}

template <>
struct AccumTCast<bf16, f32> {
  static constexpr bool supported = true;
  __device__ static inline bf16 cast(f32 val) { return f32_to_bf16(val); }
};

template <>
struct AccumTCast<f32, bf16> {
  static constexpr bool supported = true;
  __device__ static inline f32 cast(bf16 val) { return bf16_to_f32(val); }
};

  #ifdef __CHOREO_TARGET_NATIVE_FP8_SUPPORT__
template <>
struct AccumTCast<f8_e4m3, f32> {
  static constexpr bool supported = true;
  __device__ static inline f8_e4m3 cast(f32 val) { return f8_e4m3(val); }
};

template <>
struct AccumTCast<f32, f8_e4m3> {
  static constexpr bool supported = true;
  __device__ static inline f32 cast(f8_e4m3 val) { return float(val); }
};

template <>
struct AccumTCast<f8_e5m2, f32> {
  static constexpr bool supported = true;
  __device__ static inline f8_e5m2 cast(f32 val) { return f8_e5m2(val); }
};

template <>
struct AccumTCast<f32, f8_e5m2> {
  static constexpr bool supported = true;
  __device__ static inline f32 cast(f8_e5m2 val) { return float(val); }
};
  #endif

// --------------- load A policies ---------------
struct Policy_A_M8N8K4 {
  template <class Tensor>
  __device__ static auto load(Tensor const& A) {
    int lane = threadIdx.x & 31;
    using value_type = typename Tensor::value_type;
    if constexpr (std::is_same<value_type, f16>::value) {
      int row;
      if (lane < 16)
        row = lane & 3;
      else
        row = (lane & 3) + 4;
      auto A_u32 = cute::recast<uint32_t>(A);
      uint32_t a0 = A_u32(row, 0);
      uint32_t a1 = A_u32(row, 1);
      return cutlass::Array<uint32_t, 2>{a0, a1};
    } else if constexpr (std::is_same<value_type, double>::value) {
      int row = lane >> 2;
      int col = lane & 3;
      double a0 = A(row, col);
      return cutlass::Array<double, 1>{a0};
    } else {
      static_assert(sizeof(Tensor) != sizeof(Tensor),
                    "unsupported data type in this MMA");
    }
  }
};

// for s8, u8
struct Policy_A_M8N8K16 {
  template <class Tensor>
  __device__ static auto load(Tensor const& A) {
    int lane = threadIdx.x & 31;
    int gid = lane >> 2;
    int tid_in_group = lane & 3;
    int row = gid;
    int col = tid_in_group * 4;
    uint32_t a0 = 0;
      // TODO: if use recast, res error
  #pragma unroll
    for (int i = 3; i >= 0; i--) a0 = (a0 << 8) | bitcast_u32(A(row, col + i));
    return cutlass::Array<uint32_t, 1>{a0};
  }
};

// for s4, u4
struct Policy_A_M8N8K32 {
  template <class Tensor>
  __device__ static auto load(Tensor const& A) {
    int lane = threadIdx.x & 31;
    int gid = lane >> 2;
    int tid_in_group = lane & 3;
    // TODO
  }
};

// for b1
struct Policy_A_M8N8K128 {
  template <class Tensor>
  __device__ static auto load(Tensor const& A) {
    int lane = threadIdx.x & 31;
    int gid = lane >> 2;
    int tid_in_group = lane & 3;
    // TODO
  }
};

// for tf32
struct Policy_A_M16N8K4 {
  template <class Tensor>
  __device__ static auto load(Tensor const& A) {
    int lane = threadIdx.x & 31;
    int gid = lane >> 2;
    int tid_in_group = lane & 3;
    int row0 = gid;
    int row1 = gid + 8;
    int col = tid_in_group;
    using value_type = typename Tensor::value_type;
    if constexpr (std::is_same<value_type, tf32>::value ||
                  std::is_same<value_type, float>::value) {
      auto A_u32 = cute::recast<uint32_t>(A);
      uint32_t a0 = A_u32(row0, col);
      uint32_t a1 = A_u32(row1, col);
      return cutlass::Array<uint32_t, 2>{a0, a1};
    } else if constexpr (std::is_same<value_type, double>::value) {
      double a0 = A(row0, col);
      double a1 = A(row1, col);
      return cutlass::Array<double, 2>{a0, a1};
    } else {
      static_assert(sizeof(Tensor) != sizeof(Tensor),
                    "unsupported data type in this MMA policy");
    }
  }
};

struct Policy_A_M16N8K8 {
  template <class Tensor>
  __device__ static auto load(Tensor const& A) {
    int lane = threadIdx.x & 31;
    int gid = lane >> 2;
    int tid_in_group = lane & 3;
    int row0 = gid;
    int row1 = gid + 8;
    int col0 = tid_in_group;
    int col1 = tid_in_group + 4;
    using value_type = typename Tensor::value_type;
    if constexpr (std::is_same<value_type, f16>::value ||
                  std::is_same<value_type, bf16>::value) {
      auto A_u32 = cute::recast<uint32_t>(A);
      uint32_t a0 = A_u32(row0, col0);
      uint32_t a1 = A_u32(row1, col0);
      return cutlass::Array<uint32_t, 2>{a0, a1};
    } else if constexpr (std::is_same<value_type, tf32>::value ||
                         std::is_same<value_type, float>::value) {
      auto A_u32 = cute::recast<uint32_t>(A);
      uint32_t a0 = A_u32(row0, col0);
      uint32_t a1 = A_u32(row1, col0);
      uint32_t a2 = A_u32(row0, col1);
      uint32_t a3 = A_u32(row1, col1);
      return cutlass::Array<uint32_t, 4>{a0, a1, a2, a3};
    } else if constexpr (std::is_same<value_type, double>::value) {
      double a0 = A(row0, col0);
      double a1 = A(row1, col0);
      double a2 = A(row0, col1);
      double a3 = A(row1, col1);
      return cutlass::Array<double, 4>{a0, a1, a2, a3};
    } else {
      static_assert(sizeof(Tensor) != sizeof(Tensor),
                    "unsupported data type in this MMA policy");
    }
  }
};

struct Policy_A_M16N8K16 {
  template <class Tensor>
  __device__ static auto load(Tensor const& A) {
    int lane = threadIdx.x & 31;
    int gid = lane >> 2;
    int tid_in_group = lane & 3;
    int row0 = gid, row1 = gid + 8;

    using value_type = typename Tensor::value_type;
    if constexpr (std::is_same<value_type, double>::value) {
      int col0 = tid_in_group, col1 = tid_in_group + 4, col2 = tid_in_group + 8,
          col3 = tid_in_group + 12;
      double a0 = A(row0, col0);
      double a1 = A(row1, col0);
      double a2 = A(row0, col1);
      double a3 = A(row1, col1);
      double a4 = A(row0, col2);
      double a5 = A(row1, col2);
      double a6 = A(row0, col3);
      double a7 = A(row1, col3);
      return cutlass::Array<double, 8>{a0, a1, a2, a3, a4, a5, a6, a7};
    } else if constexpr (std::is_same<value_type, f16>::value ||
                         std::is_same<value_type, bf16>::value) {
      auto A_u32 = cute::recast<uint32_t>(A);
      int col0 = tid_in_group, col1 = tid_in_group + 4;
      uint32_t a0 = A_u32(row0, col0);
      uint32_t a1 = A_u32(row1, col0);
      uint32_t a2 = A_u32(row0, col1);
      uint32_t a3 = A_u32(row1, col1);
      return cutlass::Array<uint32_t, 4>{a0, a1, a2, a3};
    } else if constexpr (std::is_same<value_type, uint8_t>::value ||
                         std::is_same<value_type, int8_t>::value ||
                         std::is_same<value_type, f8_e4m3>::value ||
                         std::is_same<value_type, f8_e5m2>::value) {
      int col = tid_in_group * 4;
      uint32_t a0 = 0;
  #pragma unroll
      for (int i = 3; i >= 0; i--)
        a0 = (a0 << 8) | bitcast_u32(A(row0, col + i));
      uint32_t a1 = 0;
  #pragma unroll
      for (int i = 3; i >= 0; i--)
        a1 = (a1 << 8) | bitcast_u32(A(row1, col + i));
      return cutlass::Array<uint32_t, 2>{a0, a1};
    } else {
      static_assert(sizeof(Tensor) != sizeof(Tensor),
                    "unsupported data type in this MMA policy");
    }
  }
};

struct Policy_A_M16N8K32 {
  template <class Tensor>
  __device__ static auto load(Tensor const& A) {
    int lane = threadIdx.x & 31;
    int gid = lane >> 2;
    int tid_in_group = lane & 3;
    int row0 = gid;
    int row1 = gid + 8;
    using value_type = typename Tensor::value_type;
    if constexpr (std::is_same<value_type, f8_e4m3>::value ||
                  std::is_same<value_type, f8_e5m2>::value) {
      int col0 = tid_in_group;
      int col1 = tid_in_group + 4;
      auto A_u32 = cute::recast<uint32_t>(A);
      uint32_t a0 = A_u32(row0, col0);
      uint32_t a1 = A_u32(row1, col0);
      uint32_t a2 = A_u32(row0, col1);
      uint32_t a3 = A_u32(row1, col1);
      return cutlass::Array<uint32_t, 4>{a0, a1, a2, a3};
    } else if constexpr (std::is_same<value_type, uint8_t>::value ||
                         std::is_same<value_type, int8_t>::value) {
      int col0 = tid_in_group * 4;
      int col1 = tid_in_group * 4 + 16;
      uint32_t a0 = 0;
  #pragma unroll
      for (int i = 3; i >= 0; i--)
        a0 = (a0 << 8) | bitcast_u32(A(row0, col0 + i));
      uint32_t a1 = 0;
  #pragma unroll
      for (int i = 3; i >= 0; i--)
        a1 = (a1 << 8) | bitcast_u32(A(row1, col0 + i));
      uint32_t a2 = 0;
  #pragma unroll
      for (int i = 3; i >= 0; i--)
        a2 = (a2 << 8) | bitcast_u32(A(row0, col1 + i));
      uint32_t a3 = 0;
  #pragma unroll
      for (int i = 3; i >= 0; i--)
        a3 = (a3 << 8) | bitcast_u32(A(row1, col1 + i));
      return cutlass::Array<uint32_t, 4>{a0, a1, a2, a3};
    }
  }
};

// Sparse A load policy for m16n8k16 (compressed K/2 = 8 elements per row)
// Each thread loads 2 half values from the compressed sparse matrix
struct Policy_A_Sparse_M16N8K16 {
  template <class Tensor>
  __device__ static auto load(Tensor const& A) {
    int lane = threadIdx.x & 31;
    int group_id = lane >> 2;   // 0..7
    int thread_id = lane & 0x3; // 0..3

    using value_type = typename Tensor::value_type;
    static_assert(std::is_same<value_type, f16>::value ||
                      std::is_same<value_type, bf16>::value,
                  "Sparse m16n8k16 only supports f16/bf16");

    // For 2:4 sparse m16n8k16: A is [16, 8] (compressed from [16, 16])
    // Each thread loads 2 values packed into one uint32_t
    int col_base = thread_id * 2; // 0, 2, 4, 6
    uint32_t a0 = (bitcast_u32(A(group_id, col_base + 1)) << 16) |
                  bitcast_u32(A(group_id, col_base));
    uint32_t a1 = (bitcast_u32(A(group_id + 8, col_base + 1)) << 16) |
                  bitcast_u32(A(group_id + 8, col_base));
    return cutlass::Array<uint32_t, 2>{a0, a1};
  }
};

// Sparse A load policy for m16n8k32 (compressed K/2 = 16 elements per row)
// Each thread loads 4 half values from the compressed sparse matrix
struct Policy_A_Sparse_M16N8K32 {
  template <class Tensor>
  __device__ static auto load(Tensor const& A) {
    int lane = threadIdx.x & 31;
    int group_id = lane >> 2;   // 0..7
    int thread_id = lane & 0x3; // 0..3

    using value_type = typename Tensor::value_type;
    static_assert(std::is_same<value_type, f16>::value ||
                      std::is_same<value_type, bf16>::value,
                  "Sparse m16n8k32 only supports f16/bf16");

    // For 2:4 sparse m16n8k32: A is [16, 16] (compressed from [16, 32])
    // Fragment layout requires 8 half values -> 4 uint32_t registers
    // Following the sptc-demo pattern for manual lane loading
    value_type vals[8];
  #pragma unroll
    for (int ai = 0; ai < 8; ++ai) {
      int row = (ai < 2 || (ai >= 4 && ai < 6)) ? group_id : (group_id + 8);
      int col_base = (ai < 4) ? (thread_id * 4) : (thread_id * 4 + 16);
      int chunk = col_base / 4;
      int val_idx = ai & 0x1;
      // A is [16, 16] compressed; chunk * 2 + val_idx gives column
      vals[ai] = A(row, chunk * 2 + val_idx);
    }
    uint32_t p0 = (bitcast_u32(vals[1]) << 16) | bitcast_u32(vals[0]);
    uint32_t p1 = (bitcast_u32(vals[3]) << 16) | bitcast_u32(vals[2]);
    uint32_t p2 = (bitcast_u32(vals[5]) << 16) | bitcast_u32(vals[4]);
    uint32_t p3 = (bitcast_u32(vals[7]) << 16) | bitcast_u32(vals[6]);
    return cutlass::Array<uint32_t, 4>{p0, p1, p2, p3};
  }
};

// Sparse A load policy for m16n8k64 (compressed K/2 = 32 elements per row)
// Each thread loads 16 fp8 values -> 4 uint32_t registers
struct Policy_A_Sparse_M16N8K64 {
  template <class Tensor>
  __device__ static auto load(Tensor const& A) {
    int lane = threadIdx.x & 31;
    int group_id = lane >> 2;   // 0..7
    int thread_id = lane & 0x3; // 0..3

    using value_type = typename Tensor::value_type;
    static_assert(std::is_same<value_type, f8_e4m3>::value ||
                      std::is_same<value_type, f8_e5m2>::value,
                  "Sparse m16n8k64 only supports fp8 types");

    // For 2:4 sparse m16n8k64: A is [16, 32] (compressed from [16, 64])
    // Recast to uint32 (4 fp8 per reg) and follow dense K32 layout
    auto A_u32 = cute::recast<uint32_t>(A);
    int col0 = thread_id;     // 0..3
    int col1 = thread_id + 4; // 4..7
    uint32_t a0 = A_u32(group_id, col0);
    uint32_t a1 = A_u32(group_id + 8, col0);
    uint32_t a2 = A_u32(group_id, col1);
    uint32_t a3 = A_u32(group_id + 8, col1);
    return cutlass::Array<uint32_t, 4>{a0, a1, a2, a3};
  }
};

// Sparse metadata load policy for m16n8k32
struct Policy_E_Sparse_M16N8K32 {
  template <class Tensor>
  __device__ static uint32_t load(Tensor const& E) {
    int lane = threadIdx.x & 31;
    int group_id = lane >> 2; // 0..7
    int thread_id = lane & 3; // 0..3
    if (thread_id == 0 || thread_id == 1) {
      uint32_t e0 = E(group_id, 0);
      uint32_t e1 = E(group_id + 8, 0);
      uint32_t lo0 = e0 & 0xFFFFu;
      uint32_t hi0 = (e0 >> 16) & 0xFFFFu;
      uint32_t lo1 = e1 & 0xFFFFu;
      uint32_t hi1 = (e1 >> 16) & 0xFFFFu;
      if (thread_id == 0) return (lo1 << 16) | lo0;
      return (hi1 << 16) | hi0;
    }
    return 0;
  }
};

// Sparse metadata load policy for m16n8k16
struct Policy_E_Sparse_M16N8K16 {
  template <class Tensor>
  __device__ static uint32_t load(Tensor const& E) {
    int lane = threadIdx.x & 31;
    int group_id = lane >> 2; // 0..7
    uint32_t lo = E(group_id, 0) & 0xFFFFu;
    uint32_t hi = E(group_id + 8, 0) & 0xFFFFu;
    return (hi << 16) | lo;
  }
};

// Sparse metadata load policy for m16n8k64 (FP8 path)
// Map the two 32-bit metadata words for each row pair across the 4 lanes.
struct Policy_E_Sparse_M16N8K64 {
  template <class Tensor>
  __device__ static uint32_t load(Tensor const& E) {
    int lane = threadIdx.x & 31;
    int group_id = lane >> 2;   // 0..7
    int thread_id = lane & 0x3; // 0..3

    uint32_t row0_lo = E(group_id, 0);
    uint32_t row0_hi = E(group_id, 1);
    uint32_t row1_lo = E(group_id + 8, 0);
    uint32_t row1_hi = E(group_id + 8, 1);

    if (thread_id == 0) return row0_lo;
    if (thread_id == 1) return row1_lo;
    if (thread_id == 2) return row0_hi;
    return row1_hi;
  }
};

// for s4, u4 and e2m1
struct Policy_A_M16N8K64 {

  template <class Tensor>
  __device__ static auto load(Tensor const& A) {
    int lane = threadIdx.x & 31;
    int gid = lane >> 2;
    int tid_in_group = lane & 3;
    // TODO
  }
};

// for b1
struct Policy_A_M16N8K128 {
  template <class Tensor>
  __device__ static auto load(Tensor const& A) {
    int lane = threadIdx.x & 31;
    int gid = lane >> 2;
    int tid_in_group = lane & 3;
    // TODO
  }
};

// for b1
struct Policy_A_M16N8K256 {
  template <class Tensor>
  __device__ static auto load(Tensor const& A) {
    int lane = threadIdx.x & 31;
    int gid = lane >> 2;
    int tid_in_group = lane & 3;
    // TODO
  }
};

// --------------- load B policies ---------------
struct Policy_B_M8N8K4 {
  template <class Tensor>
  __device__ static auto load(Tensor const& B) {
    int lane = threadIdx.x & 31;
    using value_type = typename Tensor::value_type;
    if constexpr (std::is_same<value_type, f16>::value) {
      int col;
      if (lane < 16)
        col = lane & 3;
      else
        col = (lane & 3) + 4;
      uint32_t b0 = (bitcast_u32(B(1, col)) << 16) | bitcast_u32(B(0, col));
      uint32_t b1 = (bitcast_u32(B(3, col)) << 16) | bitcast_u32(B(2, col));
      return cutlass::Array<uint32_t, 2>{b0, b1};
    } else if constexpr (std::is_same<value_type, double>::value) {
      int row = lane & 3;
      int col = lane >> 2;
      double b0 = B(row, col);
      return cutlass::Array<double, 1>{b0};
    } else {
      static_assert(sizeof(Tensor) != sizeof(Tensor),
                    "unsupported data type in this MMA");
    }
  }
};

// for s8, u8
struct Policy_B_M8N8K16 {
  template <class Tensor>
  __device__ static auto load(Tensor const& B) {
    int lane = threadIdx.x & 31;
    int gid = lane >> 2;
    int tid_in_group = lane & 3;
    int row = tid_in_group * 4;
    int col = gid;
    uint32_t b0 = 0;
  #pragma unroll
    for (int i = 3; i >= 0; i--) b0 = (b0 << 8) | bitcast_u32(B(row + i, col));
    return cutlass::Array<uint32_t, 1>{b0};
  }
};

// for s4, u4
struct Policy_B_M8N8K32 {
  template <class Tensor>
  __device__ static auto load(Tensor const& B) {
    int lane = threadIdx.x & 31;
    int gid = lane >> 2;
    int tid_in_group = lane & 3;
    // TODO
  }
};

// for b1
struct Policy_B_M8N8K128 {
  template <class Tensor>
  __device__ static auto load(Tensor const& B) {
    int lane = threadIdx.x & 31;
    int gid = lane >> 2;
    int tid_in_group = lane & 3;
    // TODO
  }
};

struct Policy_B_M16N8K4 {
  template <class Tensor>
  __device__ static auto load(Tensor const& B) {
    int lane = threadIdx.x & 31;
    int gid = lane >> 2;
    int tid_in_group = lane & 3;
    int row = tid_in_group;
    int col = gid;
    using value_type = typename Tensor::value_type;
    if constexpr (std::is_same<value_type, tf32>::value ||
                  std::is_same<value_type, float>::value) {
      uint32_t b0 = bitcast_u32(B(row, col));
      return cutlass::Array<uint32_t, 1>{b0};
    } else if constexpr (std::is_same<value_type, double>::value) {
      double b0_d = B(row, col);
      return cutlass::Array<double, 1>{b0_d};
    } else {
      static_assert(sizeof(Tensor) != sizeof(Tensor),
                    "unsupported data type in this MMA policy");
    }
  }
};

struct Policy_B_M16N8K8 {
  template <class Tensor>
  __device__ static auto load(Tensor const& B) {
    int lane = threadIdx.x & 31;
    int gid = lane >> 2;
    int tid_in_group = lane & 3;
    int row0 = tid_in_group;
    int row1 = tid_in_group + 4;
    int col = gid;
    using value_type = typename Tensor::value_type;
    if constexpr (std::is_same<value_type, f16>::value ||
                  std::is_same<value_type, bf16>::value) {
      int row = tid_in_group * 2;
      uint32_t b0 =
          (bitcast_u32(B(row + 1, col)) << 16) | bitcast_u32(B(row, col));
      return cutlass::Array<uint32_t, 1>{b0};
    } else if constexpr (std::is_same<value_type, tf32>::value ||
                         std::is_same<value_type, float>::value) {
      auto B_u32 = cute::recast<uint32_t>(B);
      uint32_t b0 = B_u32(row0, col);
      uint32_t b1 = B_u32(row1, col);
      return cutlass::Array<uint32_t, 2>{b0, b1};
    } else if constexpr (std::is_same<value_type, double>::value) {
      double b0 = B(row0, col);
      double b1 = B(row1, col);
      return cutlass::Array<double, 2>{b0, b1};
    } else {
      static_assert(sizeof(Tensor) != sizeof(Tensor),
                    "unsupported data type in this MMA policy");
    }
  }
};

struct Policy_B_M16N8K16 {
  template <class Tensor>
  __device__ static auto load(Tensor const& B) {
    int lane = threadIdx.x & 31;
    int gid = lane >> 2;
    int tid_in_group = lane & 3;

    using value_type = typename Tensor::value_type;
    if constexpr (std::is_same<value_type, double>::value) {
      int row = tid_in_group;
      int col = gid;
      double b0 = B(row, col);
      double b1 = B(row + 4, col);
      double b2 = B(row + 8, col);
      double b3 = B(row + 12, col);
      return cutlass::Array<double, 4>{b0, b1, b2, b3};
    } else if constexpr (std::is_same<value_type, f16>::value ||
                         std::is_same<value_type, bf16>::value) {
      int row0 = tid_in_group * 2;
      int row1 = tid_in_group * 2 + 8;
      int col = gid;
      uint32_t b0 =
          (bitcast_u32(B(row0 + 1, col)) << 16) | bitcast_u32(B(row0, col));
      uint32_t b1 =
          (bitcast_u32(B(row1 + 1, col)) << 16) | bitcast_u32(B(row1, col));
      return cutlass::Array<uint32_t, 2>{b0, b1};
    } else if constexpr (std::is_same<value_type, uint8_t>::value ||
                         std::is_same<value_type, int8_t>::value ||
                         std::is_same<value_type, f8_e4m3>::value ||
                         std::is_same<value_type, f8_e5m2>::value) {
      int row = tid_in_group * 4;
      int col = gid;
      uint32_t b0 = 0;
  #pragma unroll
      for (int i = 3; i >= 0; i--)
        b0 = (b0 << 8) | bitcast_u32(B(row + i, col));
      return cutlass::Array<uint32_t, 1>{b0};
    } else {
      static_assert(sizeof(Tensor) != sizeof(Tensor),
                    "unsupported data type in this MMA policy");
    }
  }
};

// Sparse B policy for m16n8k16 with K-major [N, K] layout
struct Policy_B_Sparse_M16N8K16 {
  template <class Tensor>
  __device__ static auto load(Tensor const& B) {
    int lane = threadIdx.x & 31;
    int gid = lane >> 2;
    int tid_in_group = lane & 3;

    using value_type = typename Tensor::value_type;
    if constexpr (std::is_same<value_type, double>::value) {
      int row = tid_in_group;
      int col = gid;
      double b0 = B(col, row);
      double b1 = B(col, row + 4);
      double b2 = B(col, row + 8);
      double b3 = B(col, row + 12);
      return cutlass::Array<double, 4>{b0, b1, b2, b3};
    } else if constexpr (std::is_same<value_type, f16>::value ||
                         std::is_same<value_type, bf16>::value) {
      int row0 = tid_in_group * 2;
      int row1 = tid_in_group * 2 + 8;
      int col = gid;
      uint32_t b0 =
          (bitcast_u32(B(col, row0 + 1)) << 16) | bitcast_u32(B(col, row0));
      uint32_t b1 =
          (bitcast_u32(B(col, row1 + 1)) << 16) | bitcast_u32(B(col, row1));
      return cutlass::Array<uint32_t, 2>{b0, b1};
    } else if constexpr (std::is_same<value_type, uint8_t>::value ||
                         std::is_same<value_type, int8_t>::value ||
                         std::is_same<value_type, f8_e4m3>::value ||
                         std::is_same<value_type, f8_e5m2>::value) {
      int row = tid_in_group * 4;
      int col = gid;
      uint32_t b0 = 0;
  #pragma unroll
      for (int i = 3; i >= 0; i--)
        b0 = (b0 << 8) | bitcast_u32(B(col, row + i));
      return cutlass::Array<uint32_t, 1>{b0};
    } else {
      static_assert(sizeof(Tensor) != sizeof(Tensor),
                    "unsupported data type in this MMA policy");
    }
  }
};

// for s8, u8, e4m3, e5m2, e3m2, e2m3, e2m1, and also f16/bf16 for sparse MMA
struct Policy_B_M16N8K32 {
  template <class Tensor>
  __device__ static auto load(Tensor const& B) {
    int lane = threadIdx.x & 31;
    int gid = lane >> 2;
    int tid_in_group = lane % 4;

    int row0 = tid_in_group * 4;
    int row1 = tid_in_group * 4 + 16;
    int col = gid;
    if constexpr (std::is_same<typename Tensor::value_type, f16>::value ||
                  std::is_same<typename Tensor::value_type, bf16>::value) {
      // For f16/bf16 m16n8k32: B is [32, 8], need 4 registers
      int row2 = tid_in_group * 4 + 8;
      uint32_t b0 =
          (bitcast_u32(B(row0 + 1, col)) << 16) | bitcast_u32(B(row0, col));
      uint32_t b1 =
          (bitcast_u32(B(row0 + 3, col)) << 16) | bitcast_u32(B(row0 + 2, col));
      uint32_t b2 =
          (bitcast_u32(B(row2 + 1, col)) << 16) | bitcast_u32(B(row2, col));
      uint32_t b3 =
          (bitcast_u32(B(row2 + 3, col)) << 16) | bitcast_u32(B(row2 + 2, col));
      return cutlass::Array<uint32_t, 4>{b0, b1, b2, b3};
    } else if constexpr (std::is_same<typename Tensor::value_type, s8>::value ||
                         std::is_same<typename Tensor::value_type, u8>::value ||
                         std::is_same<typename Tensor::value_type,
                                      f8_e4m3>::value ||
                         std::is_same<typename Tensor::value_type,
                                      f8_e5m2>::value) {
      uint32_t b0 = (bitcast_u32(B(row0 + 3, col)) << 24) |
                    (bitcast_u32(B(row0 + 2, col)) << 16) |
                    (bitcast_u32(B(row0 + 1, col)) << 8) |
                    bitcast_u32(B(row0, col));
      uint32_t b1 = (bitcast_u32(B(row1 + 3, col)) << 24) |
                    (bitcast_u32(B(row1 + 2, col)) << 16) |
                    (bitcast_u32(B(row1 + 1, col)) << 8) |
                    bitcast_u32(B(row1, col));

      return cutlass::Array<uint32_t, 2>{b0, b1};
    } else {
      static_assert(sizeof(Tensor) != sizeof(Tensor),
                    "unsupported data type in this MMA policy");
    }
  }
};

// Sparse row.row path expects B in K-major (shape [N, K]).
// This policy swaps B indices for f16/bf16 to match that layout.
struct Policy_B_Sparse_M16N8K32 {
  template <class Tensor>
  __device__ static auto load(Tensor const& B) {
    int lane = threadIdx.x & 31;
    int gid = lane >> 2;
    int tid_in_group = lane % 4;

    int row0 = tid_in_group * 2;
    int col = gid;
    if constexpr (std::is_same<typename Tensor::value_type, f16>::value ||
                  std::is_same<typename Tensor::value_type, bf16>::value) {
      // Swap indices: B is [N, K]
      uint32_t b0 =
          (bitcast_u32(B(col, row0 + 1)) << 16) | bitcast_u32(B(col, row0));
      uint32_t b1 =
          (bitcast_u32(B(col, row0 + 9)) << 16) | bitcast_u32(B(col, row0 + 8));
      uint32_t b2 = (bitcast_u32(B(col, row0 + 17)) << 16) |
                    bitcast_u32(B(col, row0 + 16));
      uint32_t b3 = (bitcast_u32(B(col, row0 + 25)) << 16) |
                    bitcast_u32(B(col, row0 + 24));
      return cutlass::Array<uint32_t, 4>{b0, b1, b2, b3};
    } else {
      return Policy_B_M16N8K32::load(B);
    }
  }
};

// Sparse B policy for m16n8k64 with K-major [N, K] layout (fp8 path)
struct Policy_B_Sparse_M16N8K64 {
  template <class Tensor>
  __device__ static auto load(Tensor const& B) {
    int lane = threadIdx.x & 31;
    int gid = lane >> 2;
    int tid_in_group = lane & 3;

    using value_type = typename Tensor::value_type;
    static_assert(std::is_same<value_type, f8_e4m3>::value ||
                      std::is_same<value_type, f8_e5m2>::value,
                  "Sparse m16n8k64 only supports fp8 types");

    // Swap indices: B is [N, K]
    int col = gid;
    int row0 = tid_in_group * 4;
    int row1 = tid_in_group * 4 + 16;
    int row2 = tid_in_group * 4 + 32;
    int row3 = tid_in_group * 4 + 48;
    uint32_t b0 = (bitcast_u32(B(col, row0 + 3)) << 24) |
                  (bitcast_u32(B(col, row0 + 2)) << 16) |
                  (bitcast_u32(B(col, row0 + 1)) << 8) |
                  bitcast_u32(B(col, row0));
    uint32_t b1 = (bitcast_u32(B(col, row1 + 3)) << 24) |
                  (bitcast_u32(B(col, row1 + 2)) << 16) |
                  (bitcast_u32(B(col, row1 + 1)) << 8) |
                  bitcast_u32(B(col, row1));
    uint32_t b2 = (bitcast_u32(B(col, row2 + 3)) << 24) |
                  (bitcast_u32(B(col, row2 + 2)) << 16) |
                  (bitcast_u32(B(col, row2 + 1)) << 8) |
                  bitcast_u32(B(col, row2));
    uint32_t b3 = (bitcast_u32(B(col, row3 + 3)) << 24) |
                  (bitcast_u32(B(col, row3 + 2)) << 16) |
                  (bitcast_u32(B(col, row3 + 1)) << 8) |
                  bitcast_u32(B(col, row3));
    return cutlass::Array<uint32_t, 4>{b0, b1, b2, b3};
  }
};

// for s4, u4 and e2m1
struct Policy_B_M16N8K64 {
  template <class Tensor>
  __device__ static auto load(Tensor const& B) {
    int lane = threadIdx.x & 31;
    int gid = lane >> 2;
    int tid_in_group = lane & 3;
    // TODO
  }
};

// for b1
struct Policy_B_M16N8K128 {
  template <class Tensor>
  __device__ static auto load(Tensor const& B) {
    int lane = threadIdx.x & 31;
    int gid = lane >> 2;
    int tid_in_group = lane & 3;
    // TODO
  }
};

// for b1
struct Policy_B_M16N8K256 {
  template <class Tensor>
  __device__ static auto load(Tensor const& B) {
    int lane = threadIdx.x & 31;
    int gid = lane >> 2;
    int tid_in_group = lane & 3;
    // TODO
  }
};

// ------------------- store/load D fragment -------------------
struct Policy_D_M8N8 {
  template <class Tensor, class AccumT>
  __device__ static void store(Tensor& D, AccumT const* d) {
    int lane = threadIdx.x & 31;
    using value_type = typename Tensor::value_type;
    if constexpr (std::is_same<value_type, double>::value ||
                  std::is_same<value_type, s32>::value) {
      static_assert(std::is_same<AccumT, value_type>::value,
                    "AccumT must be same as value_type");
      int gid = lane >> 2;
      int tid_in_group = lane & 3;
      int row = gid;
      int col0 = tid_in_group * 2;
      int col1 = col0 + 1;
      auto D_casted = cute::recast<AccumT>(D);
      D_casted(row, col0) = d[0];
      D_casted(row, col1) = d[1];
    } else if constexpr (std::is_same<AccumT, float>::value) {
      int row = (lane & 1);
      if (lane >= 16) row += 4;
      int col = lane & 2;
      D(row, col) = cast_if<value_type>(d[0]);
      D(row, col + 1) = cast_if<value_type>(d[1]);
      D(row + 2, col) = cast_if<value_type>(d[2]);
      D(row + 2, col + 1) = cast_if<value_type>(d[3]);
      D(row, col + 4) = cast_if<value_type>(d[4]);
      D(row, col + 4 + 1) = cast_if<value_type>(d[5]);
      D(row + 2, col + 4) = cast_if<value_type>(d[6]);
      D(row + 2, col + 4 + 1) = cast_if<value_type>(d[7]);
    } else if constexpr (std::is_same<AccumT, f16>::value) {
      static_assert(std::is_same<AccumT, value_type>::value,
                    "AccumT must be same as value_type");
      int row = (lane & 3);
      if (lane >= 16) row = row + 4;
      D(row, 0) = d[0];
      D(row, 1) = d[1];
      D(row, 2) = d[2];
      D(row, 3) = d[3];
      D(row, 4) = d[4];
      D(row, 5) = d[5];
      D(row, 6) = d[6];
      D(row, 7) = d[7];
    } else {
      static_assert(sizeof(Tensor) != sizeof(Tensor),
                    "unsupported data type in this MMA policy");
    }
  }

  template <class Tensor, class AccumT>
  __device__ static void store_trans(Tensor& D, AccumT const* d) {
    int lane = threadIdx.x & 31;
    using value_type = typename Tensor::value_type;
    if constexpr (std::is_same<value_type, double>::value ||
                  std::is_same<value_type, s32>::value) {
      static_assert(std::is_same<AccumT, value_type>::value,
                    "AccumT must be same as value_type");
      int gid = lane >> 2;
      int tid_in_group = lane & 3;
      int row = gid;
      int col0 = tid_in_group * 2;
      int col1 = col0 + 1;
      auto D_casted = cute::recast<AccumT>(D);
      D_casted(col0, row) = d[0];
      D_casted(col1, row) = d[1];
    } else if constexpr (std::is_same<AccumT, float>::value) {
      int row = (lane & 1);
      if (lane >= 16) row += 4;
      int col = lane & 2;
      D(col, row) = cast_if<value_type>(d[0]);
      D(col + 1, row) = cast_if<value_type>(d[1]);
      D(col, row + 2) = cast_if<value_type>(d[2]);
      D(col + 1, row + 2) = cast_if<value_type>(d[3]);
      D(col + 4, row) = cast_if<value_type>(d[4]);
      D(col + 4 + 1, row) = cast_if<value_type>(d[5]);
      D(col + 4, row + 2) = cast_if<value_type>(d[6]);
      D(col + 4 + 1, row + 2) = cast_if<value_type>(d[7]);
    } else if constexpr (std::is_same<AccumT, f16>::value) {
      static_assert(std::is_same<AccumT, value_type>::value,
                    "AccumT must be same as value_type");
      int row = (lane & 3);
      if (lane >= 16) row = row + 4;
      D(0, row) = d[0];
      D(1, row) = d[1];
      D(2, row) = d[2];
      D(3, row) = d[3];
      D(4, row) = d[4];
      D(5, row) = d[5];
      D(6, row) = d[6];
      D(7, row) = d[7];
    } else {
      static_assert(sizeof(Tensor) != sizeof(Tensor),
                    "unsupported data type in this MMA policy");
    }
  }

  template <class Tensor, class AccumT>
  __device__ static void load(Tensor const& D, AccumT* d) {
    int lane = threadIdx.x & 31;
    using value_type = typename Tensor::value_type;
    if constexpr (std::is_same<value_type, double>::value ||
                  std::is_same<value_type, s32>::value) {
      static_assert(std::is_same<AccumT, value_type>::value,
                    "AccumT must be same as value_type");
      int gid = lane >> 2;
      int tid_in_group = lane & 3;
      int row = gid;
      int col0 = tid_in_group * 2;
      int col1 = col0 + 1;
      auto D_casted = cute::recast<AccumT>(D);
      d[0] = D_casted(row, col0);
      d[1] = D_casted(row, col1);
    } else if constexpr (std::is_same<value_type, float>::value) {
      int row = (lane & 1);
      if (lane >= 16) row += 4;
      int col = lane & 2;
      d[0] = cast_if<AccumT>(D(row, col));
      d[1] = cast_if<AccumT>(D(row, col + 1));
      d[2] = cast_if<AccumT>(D(row + 2, col));
      d[3] = cast_if<AccumT>(D(row + 2, col + 1));
      d[4] = cast_if<AccumT>(D(row, col + 4));
      d[5] = cast_if<AccumT>(D(row, col + 4 + 1));
      d[6] = cast_if<AccumT>(D(row + 2, col + 4));
      d[7] = cast_if<AccumT>(D(row + 2, col + 4 + 1));
    } else if constexpr (std::is_same<value_type, f16>::value) {
      static_assert(std::is_same<AccumT, value_type>::value,
                    "AccumT must be same as value_type");
      int row = (lane & 3);
      if (lane >= 16) row = row + 4;
      d[0] = D(row, 0);
      d[1] = D(row, 1);
      d[2] = D(row, 2);
      d[3] = D(row, 3);
      d[4] = D(row, 4);
      d[5] = D(row, 5);
      d[6] = D(row, 6);
      d[7] = D(row, 7);
    } else {
      static_assert(sizeof(Tensor) != sizeof(Tensor),
                    "unsupported data type in this MMA policy");
    }
  }
};

struct Policy_D_M16N8 {
  template <class Tensor, class AccumT>
  __device__ static void load(Tensor const& D, AccumT* d) {
    int lane = threadIdx.x & 31;
    int gid = lane >> 2;
    int tid_in_group = lane & 3;
    int row0 = gid;
    int row1 = gid + 8;
    int col0 = tid_in_group * 2;
    int col1 = tid_in_group * 2 + 1;
    d[0] = cast_if<AccumT>(D(row0, col0));
    d[1] = cast_if<AccumT>(D(row0, col1));
    d[2] = cast_if<AccumT>(D(row1, col0));
    d[3] = cast_if<AccumT>(D(row1, col1));
  }

  template <class Tensor, class AccumT>
  __device__ static void store(Tensor& D, AccumT const* d) {
    int lane = threadIdx.x & 31;
    int gid = lane >> 2;
    int tid_in_group = lane & 3;
    int row0 = gid;
    int row1 = gid + 8;
    int col0 = tid_in_group * 2;
    int col1 = tid_in_group * 2 + 1;
    using value_type = typename Tensor::value_type;
    D(row0, col0) = cast_if<value_type>(d[0]);
    D(row0, col1) = cast_if<value_type>(d[1]);
    D(row1, col0) = cast_if<value_type>(d[2]);
    D(row1, col1) = cast_if<value_type>(d[3]);
  }

  template <class Tensor, class AccumT>
  __device__ static void store_mask_row(Tensor& D, AccumT const* d,
                                        int row_guard) {
    int lane = threadIdx.x & 31;
    int gid = lane >> 2;
    int tid_in_group = lane & 3;
    int row0 = gid;
    int row1 = gid + 8;
    int col0 = tid_in_group * 2;
    int col1 = tid_in_group * 2 + 1;
    using value_type = typename Tensor::value_type;
    if (row0 < row_guard) {
      D(row0, col0) = cast_if<value_type>(d[0]);
      D(row0, col1) = cast_if<value_type>(d[1]);
    }
    if (row1 < row_guard) {
      D(row1, col0) = cast_if<value_type>(d[2]);
      D(row1, col1) = cast_if<value_type>(d[3]);
    }
  }

  template <class Tensor, class AccumT>
  __device__ static void store_mask_col(Tensor& D, AccumT const* d,
                                        int col_guard) {
    int lane = threadIdx.x & 31;
    int gid = lane >> 2;
    int tid_in_group = lane & 3;
    int row0 = gid;
    int row1 = gid + 8;
    int col0 = tid_in_group * 2;
    int col1 = tid_in_group * 2 + 1;
    using value_type = typename Tensor::value_type;
    if (col0 < col_guard) {
      D(row0, col0) = cast_if<value_type>(d[0]);
      D(row1, col0) = cast_if<value_type>(d[2]);
    }
    if (col1 < col_guard) {
      D(row0, col1) = cast_if<value_type>(d[1]);
      D(row1, col1) = cast_if<value_type>(d[3]);
    }
  }

  template <class Tensor, class AccumT>
  __device__ static void store_mask_row_col(Tensor& D, AccumT const* d,
                                            int row_guard, int col_guard) {
    int lane = threadIdx.x & 31;
    int gid = lane >> 2;
    int tid_in_group = lane & 3;
    int row0 = gid;
    int row1 = gid + 8;
    int col0 = tid_in_group * 2;
    int col1 = tid_in_group * 2 + 1;
    using value_type = typename Tensor::value_type;
    if (row0 < row_guard) {
      if (col0 < col_guard) D(row0, col0) = cast_if<value_type>(d[0]);
      if (col1 < col_guard) D(row0, col1) = cast_if<value_type>(d[1]);
    }
    if (row1 < row_guard) {
      if (col0 < col_guard) D(row1, col0) = cast_if<value_type>(d[2]);
      if (col1 < col_guard) D(row1, col1) = cast_if<value_type>(d[3]);
    }
  }

  template <class Tensor, class AccumT>
  __device__ static void store_trans(Tensor& D, AccumT const* d) {
    int lane = threadIdx.x & 31;
    int gid = lane >> 2;
    int tid_in_group = lane & 3;
    int row0 = gid;
    int row1 = gid + 8;
    int col0 = tid_in_group * 2;
    int col1 = tid_in_group * 2 + 1;
    using value_type = typename Tensor::value_type;
    D(col0, row0) = cast_if<value_type>(d[0]);
    D(col1, row0) = cast_if<value_type>(d[1]);
    D(col0, row1) = cast_if<value_type>(d[2]);
    D(col1, row1) = cast_if<value_type>(d[3]);
  }
};

// --------------- TMA primitives (SM90+) ---------------
// TMA SWIZZLE pattern enum for CUtensorMapSwizzle
enum class TMA_Swizzle {
  NONE = 0, // No swizzle
  B32 = 1,  // 32B swizzle
  B64 = 2,  // 64B swizzle
  B128 = 3  // 128B swizzle
};

// Helper function to convert TMA_Swizzle to CUtensorMapSwizzle string
// representation
inline constexpr const char* cuda_stringify(TMA_Swizzle swizzle) {
  switch (swizzle) {
  case TMA_Swizzle::NONE:
    return "CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_NONE";
  case TMA_Swizzle::B32: return "CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_32B";
  case TMA_Swizzle::B64: return "CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_64B";
  case TMA_Swizzle::B128:
    return "CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_128B";
  default: return "CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_NONE";
  }
}

// --------------- WGMMA primitives (SM90+) ---------------
// refer to:
// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#asynchronous-warpgroup-level-leading-dimension-byte-offset
// 9.7.15.5.1.2.2. Matrix Descriptor Format
// SWIZZLE pattern enum
enum class WGMMA_Swizzle : uint64_t {
  NS = 0,  // No swizzle
  B32 = 3, // 32B swizzle
  B64 = 2, // 64B swizzle
  B128 = 1 // 128B swizzle
};

// Major order enum
enum class WGMMA_MajorOrder {
  K_MAJOR, // K dimension is major (leading)
  MN_MAJOR // M and N dimensions are major (leading)
};

// mma shape enum
enum class WGMMA_MMAShape {
  M64N64K16,
};

// helper functions to get mma property at compile time
template <WGMMA_MMAShape Shape>
__device__ constexpr int get_mma_m() {
  if constexpr (Shape == WGMMA_MMAShape::M64N64K16) return 64;
  return 0;
}

template <WGMMA_MMAShape Shape>
__device__ constexpr int get_mma_n() {
  if constexpr (Shape == WGMMA_MMAShape::M64N64K16) return 64;
  return 0;
}

template <WGMMA_MMAShape Shape>
__device__ constexpr int get_mma_k() {
  if constexpr (Shape == WGMMA_MMAShape::M64N64K16) return 16;
  return 0;
}

template <WGMMA_MajorOrder MajorOrder>
__device__ constexpr int get_trans_a() {
  if constexpr (MajorOrder == WGMMA_MajorOrder::K_MAJOR) return 0;
  if constexpr (MajorOrder == WGMMA_MajorOrder::MN_MAJOR) return 1;
  return 0;
}

template <WGMMA_MajorOrder MajorOrder>
__device__ constexpr int get_trans_b() {
  if constexpr (MajorOrder == WGMMA_MajorOrder::K_MAJOR) return 0;
  if constexpr (MajorOrder == WGMMA_MajorOrder::MN_MAJOR) return 1;
  return 0;
}

// Helper function to encode matrix descriptor
__device__ static __forceinline__ uint64_t
matrix_descriptor_encode(uint64_t x) {
  return (((x) & 0x3FFFF) >> 0x4);
}

// Unified shared memory descriptor encoding template
// Automatically determines stride and leading dimension based on major order
// and swizzle.  When LBOBytes > 0 the caller overrides the leading-byte-offset
// (needed for MN-major layouts where N spans multiple swizzle periods, e.g.
// Layout_K_SW128_Atom tiled to N=128 for bf16).
template <WGMMA_MajorOrder MajorOrder, WGMMA_Swizzle Swizzle, int LBOBytes = 0,
          typename T>
__device__ static __forceinline__ uint64_t wgmma_make_smem_desc(T* ptr) {
  uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
  uint64_t desc = 0x0000000000000000;
  desc |= matrix_descriptor_encode(addr);

  uint64_t LBO = 0;
  uint64_t SBO = 0;

  if constexpr (MajorOrder == WGMMA_MajorOrder::K_MAJOR) {
    switch (Swizzle) {
    case WGMMA_Swizzle::NS:
      LBO = 256;
      SBO = 128;
      break;
    case WGMMA_Swizzle::B32:
      LBO = 16;
      SBO = 256;
      break;
    case WGMMA_Swizzle::B64:
      LBO = 16;
      SBO = 512;
      break;
    case WGMMA_Swizzle::B128:
      LBO = 16;
      SBO = 1024;
      break;
    }
  } else { // MN_MAJOR
    switch (Swizzle) {
    case WGMMA_Swizzle::NS:
      LBO = 16;
      SBO = 128;
      break;
    case WGMMA_Swizzle::B32:
      LBO = 16;
      SBO = 256;
      break;
    case WGMMA_Swizzle::B64:
      LBO = 16;
      SBO = 512;
      break;
    case WGMMA_Swizzle::B128:
      LBO = 16;
      SBO = 1024;
      break;
    }
  }

  if constexpr (LBOBytes > 0) LBO = LBOBytes;

  desc |= matrix_descriptor_encode(LBO) << 16;
  desc |= matrix_descriptor_encode(SBO) << 32;
  desc |= static_cast<uint64_t>(Swizzle) << 62;

  return desc;
}

// WGMMA fence/sync primitives
__device__ static __forceinline__ void warpgroup_arrive() {
  #if defined(CUTE_ARCH_MMA_SM90A_ENABLED)
  asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
  #endif
}

__device__ static __forceinline__ void warpgroup_commit_batch() {
  #if defined(CUTE_ARCH_MMA_SM90A_ENABLED)
  asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
  #endif
}

template <int PD>
__device__ static __forceinline__ void warpgroup_wait() {
  #if defined(CUTE_ARCH_MMA_SM90A_ENABLED)
  static_assert(PD >= 0 && PD <= 7, "WGMMA wait: N must be in range [0, 7]");
  asm volatile("wgmma.wait_group.sync.aligned %0;\n" ::"n"(PD) : "memory");
  #endif
}

template <class T>
__device__ static __forceinline__ void warpgroup_fence_operand(T& operand) {
  asm volatile("" : "+r"(reinterpret_cast<uint32_t&>(operand))::"memory");
}

// Unified WGMMA template with automatic descriptor selection
// Template parameters:
//   - InputT: input data type (__half or __nv_bfloat16)
//   - OutputT: output data type (float or same as InputT)
//   - MajorOrderA: major order for matrix A (K_MAJOR or MN_MAJOR)
//   - SwizzleA: swizzle pattern for matrix A
//   - MajorOrderB: major order for matrix B (K_MAJOR or MN_MAJOR)
//   - SwizzleB: swizzle pattern for matrix B
template <typename InputT, typename OutputT,
          WGMMA_MajorOrder MajorOrderA = WGMMA_MajorOrder::K_MAJOR,
          WGMMA_Swizzle SwizzleA = WGMMA_Swizzle::NS,
          WGMMA_MajorOrder MajorOrderB = WGMMA_MajorOrder::K_MAJOR,
          WGMMA_Swizzle SwizzleB = WGMMA_Swizzle::NS>
__device__ static __forceinline__ void wgmma_m64n64k16(OutputT d[4][8],
                                                       InputT* sA, InputT* sB) {
  static_assert(
      std::is_same_v<InputT, __half> || std::is_same_v<InputT, __nv_bfloat16> ||
          std::is_same_v<InputT, f8_e4m3> || std::is_same_v<InputT, f8_e5m2>,
      "wgmma_m64n64k16_unified requires __half, __nv_bfloat16 or fp8 input "
      "type");
  static_assert(
      std::is_same_v<OutputT, float> || std::is_same_v<OutputT, InputT>,
      "wgmma_m64n64k16_unified requires float or same as InputT output type");

  uint64_t desc_a = wgmma_make_smem_desc<MajorOrderA, SwizzleA>(&sA[0]);
  uint64_t desc_b = wgmma_make_smem_desc<MajorOrderB, SwizzleB>(&sB[0]);
  constexpr uint64_t trans_a = get_trans_a<MajorOrderA>();
  constexpr uint64_t trans_b = get_trans_b<MajorOrderB>();

  // Determine PTX instruction based on input and output types
  if constexpr (std::is_same_v<InputT, __half> &&
                std::is_same_v<OutputT, __half>) {
  #if defined(CUTE_ARCH_MMA_SM90A_ENABLED)
    asm volatile("{\n"
                 "wgmma.mma_async.sync.aligned.m64n64k16.f16.f16.f16 "
                 "{%0,   %1,   %2,   %3,   %4,   %5,   %6,   %7,   "
                 " %8,   %9,   %10,  %11,  %12,  %13,  %14,  %15,  "
                 " %16,  %17,  %18,  %19,  %20,  %21,  %22,  %23,  "
                 " %24,  %25,  %26,  %27,  %28,  %29,  %30,  %31},"
                 " %32,"
                 " %33,"
                 " %34, %35, %36, %37, %38;\n"
                 "}\n"
                 : "+h"(*(uint16_t*)&d[0][0]), "+h"(*(uint16_t*)&d[0][1]),
                   "+h"(*(uint16_t*)&d[0][2]), "+h"(*(uint16_t*)&d[0][3]),
                   "+h"(*(uint16_t*)&d[0][4]), "+h"(*(uint16_t*)&d[0][5]),
                   "+h"(*(uint16_t*)&d[0][6]), "+h"(*(uint16_t*)&d[0][7]),
                   "+h"(*(uint16_t*)&d[1][0]), "+h"(*(uint16_t*)&d[1][1]),
                   "+h"(*(uint16_t*)&d[1][2]), "+h"(*(uint16_t*)&d[1][3]),
                   "+h"(*(uint16_t*)&d[1][4]), "+h"(*(uint16_t*)&d[1][5]),
                   "+h"(*(uint16_t*)&d[1][6]), "+h"(*(uint16_t*)&d[1][7]),
                   "+h"(*(uint16_t*)&d[2][0]), "+h"(*(uint16_t*)&d[2][1]),
                   "+h"(*(uint16_t*)&d[2][2]), "+h"(*(uint16_t*)&d[2][3]),
                   "+h"(*(uint16_t*)&d[2][4]), "+h"(*(uint16_t*)&d[2][5]),
                   "+h"(*(uint16_t*)&d[2][6]), "+h"(*(uint16_t*)&d[2][7]),
                   "+h"(*(uint16_t*)&d[3][0]), "+h"(*(uint16_t*)&d[3][1]),
                   "+h"(*(uint16_t*)&d[3][2]), "+h"(*(uint16_t*)&d[3][3]),
                   "+h"(*(uint16_t*)&d[3][4]), "+h"(*(uint16_t*)&d[3][5]),
                   "+h"(*(uint16_t*)&d[3][6]), "+h"(*(uint16_t*)&d[3][7])
                 : "l"(desc_a), "l"(desc_b), "n"(1), "n"(1), "n"(1),
                   "n"(trans_a), "n"(trans_b));
  #endif
  } else if constexpr (std::is_same_v<InputT, __half> &&
                       std::is_same_v<OutputT, float>) {
  #if defined(CUTE_ARCH_MMA_SM90A_ENABLED)
    asm volatile("{\n"
                 "wgmma.mma_async.sync.aligned.m64n64k16.f32.f16.f16 "
                 "{%0,   %1,   %2,   %3,   %4,   %5,   %6,   %7,   "
                 " %8,   %9,   %10,  %11,  %12,  %13,  %14,  %15,  "
                 " %16,  %17,  %18,  %19,  %20,  %21,  %22,  %23,  "
                 " %24,  %25,  %26,  %27,  %28,  %29,  %30,  %31},"
                 " %32,"
                 " %33,"
                 " %34, %35, %36, %37, %38;\n"
                 "}\n"
                 : "+f"(d[0][0]), "+f"(d[0][1]), "+f"(d[0][2]), "+f"(d[0][3]),
                   "+f"(d[0][4]), "+f"(d[0][5]), "+f"(d[0][6]), "+f"(d[0][7]),
                   "+f"(d[1][0]), "+f"(d[1][1]), "+f"(d[1][2]), "+f"(d[1][3]),
                   "+f"(d[1][4]), "+f"(d[1][5]), "+f"(d[1][6]), "+f"(d[1][7]),
                   "+f"(d[2][0]), "+f"(d[2][1]), "+f"(d[2][2]), "+f"(d[2][3]),
                   "+f"(d[2][4]), "+f"(d[2][5]), "+f"(d[2][6]), "+f"(d[2][7]),
                   "+f"(d[3][0]), "+f"(d[3][1]), "+f"(d[3][2]), "+f"(d[3][3]),
                   "+f"(d[3][4]), "+f"(d[3][5]), "+f"(d[3][6]), "+f"(d[3][7])
                 : "l"(desc_a), "l"(desc_b), "n"(1), "n"(1), "n"(1),
                   "n"(trans_a), "n"(trans_b));
  #endif
  } else if constexpr (std::is_same_v<InputT, __nv_bfloat16> &&
                       std::is_same_v<OutputT, __nv_bfloat16>) {
  #if defined(CUTE_ARCH_MMA_SM90A_ENABLED)
    asm volatile("{\n"
                 "wgmma.mma_async.sync.aligned.m64n64k16.bf16.bf16.bf16 "
                 "{%0,   %1,   %2,   %3,   %4,   %5,   %6,   %7,   "
                 " %8,   %9,   %10,  %11,  %12,  %13,  %14,  %15,  "
                 " %16,  %17,  %18,  %19,  %20,  %21,  %22,  %23,  "
                 " %24,  %25,  %26,  %27,  %28,  %29,  %30,  %31},"
                 " %32,"
                 " %33,"
                 " %34, %35, %36, %37, %38;\n"
                 "}\n"
                 : "+h"(*(uint16_t*)&d[0][0]), "+h"(*(uint16_t*)&d[0][1]),
                   "+h"(*(uint16_t*)&d[0][2]), "+h"(*(uint16_t*)&d[0][3]),
                   "+h"(*(uint16_t*)&d[0][4]), "+h"(*(uint16_t*)&d[0][5]),
                   "+h"(*(uint16_t*)&d[0][6]), "+h"(*(uint16_t*)&d[0][7]),
                   "+h"(*(uint16_t*)&d[1][0]), "+h"(*(uint16_t*)&d[1][1]),
                   "+h"(*(uint16_t*)&d[1][2]), "+h"(*(uint16_t*)&d[1][3]),
                   "+h"(*(uint16_t*)&d[1][4]), "+h"(*(uint16_t*)&d[1][5]),
                   "+h"(*(uint16_t*)&d[1][6]), "+h"(*(uint16_t*)&d[1][7]),
                   "+h"(*(uint16_t*)&d[2][0]), "+h"(*(uint16_t*)&d[2][1]),
                   "+h"(*(uint16_t*)&d[2][2]), "+h"(*(uint16_t*)&d[2][3]),
                   "+h"(*(uint16_t*)&d[2][4]), "+h"(*(uint16_t*)&d[2][5]),
                   "+h"(*(uint16_t*)&d[2][6]), "+h"(*(uint16_t*)&d[2][7]),
                   "+h"(*(uint16_t*)&d[3][0]), "+h"(*(uint16_t*)&d[3][1]),
                   "+h"(*(uint16_t*)&d[3][2]), "+h"(*(uint16_t*)&d[3][3]),
                   "+h"(*(uint16_t*)&d[3][4]), "+h"(*(uint16_t*)&d[3][5]),
                   "+h"(*(uint16_t*)&d[3][6]), "+h"(*(uint16_t*)&d[3][7])
                 : "l"(desc_a), "l"(desc_b), "n"(1), "n"(1), "n"(1),
                   "n"(trans_a), "n"(trans_b));
  #endif
  } else if constexpr (std::is_same_v<InputT, __nv_bfloat16> &&
                       std::is_same_v<OutputT, float>) {
  #if defined(CUTE_ARCH_MMA_SM90A_ENABLED)
    asm volatile("{\n"
                 "wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 "
                 "{%0,   %1,   %2,   %3,   %4,   %5,   %6,   %7,   "
                 " %8,   %9,   %10,  %11,  %12,  %13,  %14,  %15,  "
                 " %16,  %17,  %18,  %19,  %20,  %21,  %22,  %23,  "
                 " %24,  %25,  %26,  %27,  %28,  %29,  %30,  %31},"
                 " %32,"
                 " %33,"
                 " %34, %35, %36, %37, %38;\n"
                 "}\n"
                 : "+f"(d[0][0]), "+f"(d[0][1]), "+f"(d[0][2]), "+f"(d[0][3]),
                   "+f"(d[0][4]), "+f"(d[0][5]), "+f"(d[0][6]), "+f"(d[0][7]),
                   "+f"(d[1][0]), "+f"(d[1][1]), "+f"(d[1][2]), "+f"(d[1][3]),
                   "+f"(d[1][4]), "+f"(d[1][5]), "+f"(d[1][6]), "+f"(d[1][7]),
                   "+f"(d[2][0]), "+f"(d[2][1]), "+f"(d[2][2]), "+f"(d[2][3]),
                   "+f"(d[2][4]), "+f"(d[2][5]), "+f"(d[2][6]), "+f"(d[2][7]),
                   "+f"(d[3][0]), "+f"(d[3][1]), "+f"(d[3][2]), "+f"(d[3][3]),
                   "+f"(d[3][4]), "+f"(d[3][5]), "+f"(d[3][6]), "+f"(d[3][7])
                 : "l"(desc_a), "l"(desc_b), "n"(1), "n"(1), "n"(1),
                   "n"(trans_a), "n"(trans_b));
  #endif
  } else if constexpr ((std::is_same_v<InputT, f8_e4m3> ||
                        std::is_same_v<InputT, f8_e5m2>) &&
                       std::is_same_v<OutputT, float>) {
  #if defined(CUTE_ARCH_MMA_SM90A_ENABLED)
    asm volatile("{\n"
                 "wgmma.mma_async.sync.aligned.m64n64k16.f32.f8.f8 "
                 "{%0,   %1,   %2,   %3,   %4,   %5,   %6,   %7,   "
                 " %8,   %9,   %10,  %11,  %12,  %13,  %14,  %15,  "
                 " %16,  %17,  %18,  %19,  %20,  %21,  %22,  %23,  "
                 " %24,  %25,  %26,  %27,  %28,  %29,  %30,  %31},"
                 " %32,"
                 " %33,"
                 " %34, %35, %36, %37, %38;\n"
                 "}\n"
                 : "+f"(d[0][0]), "+f"(d[0][1]), "+f"(d[0][2]), "+f"(d[0][3]),
                   "+f"(d[0][4]), "+f"(d[0][5]), "+f"(d[0][6]), "+f"(d[0][7]),
                   "+f"(d[1][0]), "+f"(d[1][1]), "+f"(d[1][2]), "+f"(d[1][3]),
                   "+f"(d[1][4]), "+f"(d[1][5]), "+f"(d[1][6]), "+f"(d[1][7]),
                   "+f"(d[2][0]), "+f"(d[2][1]), "+f"(d[2][2]), "+f"(d[2][3]),
                   "+f"(d[2][4]), "+f"(d[2][5]), "+f"(d[2][6]), "+f"(d[2][7]),
                   "+f"(d[3][0]), "+f"(d[3][1]), "+f"(d[3][2]), "+f"(d[3][3]),
                   "+f"(d[3][4]), "+f"(d[3][5]), "+f"(d[3][6]), "+f"(d[3][7])
                 : "l"(desc_a), "l"(desc_b), "n"(1), "n"(1), "n"(1),
                   "n"(trans_a), "n"(trans_b));
  #endif
  }
}

// wgmma store d
// reference:
// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#asynchronous-warpgroup-level-matrix-shape
struct Policy_WGMMA_D_M64K16 {
  template <class Tensor, typename AccumT, int N>
  __device__ static void store_mask_row(Tensor& D, AccumT* d, int row_guard) {
    int tid = threadIdx.x % 128;
    int lane = tid % 32;
    int warp = tid / 32;
    int row0 = warp * 16 + lane / 4;
    int row1 = row0 + 8;
    int col_num = N / 8;
    using value_type = typename Tensor::value_type;
    if (row0 < row_guard) {
  #pragma unroll
      for (int c = 0; c < col_num; c++) {
        int col0 = c * 8 + (tid % 4) * 2;
        int col1 = col0 + 1;
        D(row0, col0) = cast_if<value_type>(d[c * 4]);
        D(row0, col1) = cast_if<value_type>(d[c * 4 + 1]);
      }
    }
    if (row1 < row_guard) {
  #pragma unroll
      for (int c = 0; c < col_num; c++) {
        int col0 = c * 8 + (tid % 4) * 2;
        int col1 = col0 + 1;
        D(row1, col0) = cast_if<value_type>(d[c * 4 + 2]);
        D(row1, col1) = cast_if<value_type>(d[c * 4 + 3]);
      }
    }
  }

  template <class Tensor, typename AccumT, int N>
  __device__ static void store_mask_col(Tensor& D, AccumT* d, int col_guard) {
    int tid = threadIdx.x % 128;
    int lane = tid % 32;
    int warp = tid / 32;
    int row0 = warp * 16 + lane / 4;
    int row1 = row0 + 8;
    int col_num = N / 8;
    using value_type = typename Tensor::value_type;
  #pragma unroll
    for (int c = 0; c < col_num; c++) {
      int col0 = c * 8 + (tid % 4) * 2;
      int col1 = col0 + 1;
      if (col0 < col_guard) {
        D(row0, col0) = cast_if<value_type>(d[c * 4]);
        D(row1, col0) = cast_if<value_type>(d[c * 4 + 2]);
      }
      if (col1 < col_guard) {
        D(row0, col1) = cast_if<value_type>(d[c * 4 + 1]);
        D(row1, col1) = cast_if<value_type>(d[c * 4 + 3]);
      }
    }
  }

  template <class Tensor, typename AccumT, int N>
  __device__ static void store_mask_row_col(Tensor& D, AccumT* d, int row_guard,
                                            int col_guard) {
    int tid = threadIdx.x % 128;
    int lane = tid % 32;
    int warp = tid / 32;
    int row0 = warp * 16 + lane / 4;
    int row1 = row0 + 8;
    int col_num = N / 8;
    using value_type = typename Tensor::value_type;
    if (row0 < row_guard) {
  #pragma unroll
      for (int c = 0; c < col_num; c++) {
        int col0 = c * 8 + (tid % 4) * 2;
        int col1 = col0 + 1;
        if (col0 < col_guard) D(row0, col0) = cast_if<value_type>(d[c * 4]);
        if (col1 < col_guard) D(row0, col1) = cast_if<value_type>(d[c * 4 + 1]);
      }
    }
    if (row1 < row_guard) {
  #pragma unroll
      for (int c = 0; c < col_num; c++) {
        int col0 = c * 8 + (tid % 4) * 2;
        int col1 = col0 + 1;
        if (col0 < col_guard) D(row1, col0) = cast_if<value_type>(d[c * 4 + 2]);
        if (col1 < col_guard) D(row1, col1) = cast_if<value_type>(d[c * 4 + 3]);
      }
    }
  }

  template <class Tensor, typename AccumT, int N>
  __device__ static void store(Tensor& D, AccumT* d) {
    int tid = threadIdx.x % 128;
    int lane = tid % 32;
    int warp = tid / 32;
    int row0 = warp * 16 + lane / 4;
    int row1 = row0 + 8;
    int col_num = N / 8;
    using value_type = typename Tensor::value_type;
  #pragma unroll
    for (int c = 0; c < col_num; c++) {
      int col0 = c * 8 + (tid % 4) * 2;
      int col1 = col0 + 1;
      D(row0, col0) = cast_if<value_type>(d[c * 4]);
      D(row0, col1) = cast_if<value_type>(d[c * 4 + 1]);
      D(row1, col0) = cast_if<value_type>(d[c * 4 + 2]);
      D(row1, col1) = cast_if<value_type>(d[c * 4 + 3]);
    }
  }

  template <class Tensor, typename AccumT, int N>
  __device__ static void store_trans(Tensor& D, AccumT* d) {
    int tid = threadIdx.x % 128;
    int lane = tid % 32;
    int warp = tid / 32;
    int row0 = warp * 16 + lane / 4;
    int row1 = row0 + 8;
    int col_num = N / 8;
    using value_type = typename Tensor::value_type;
  #pragma unroll
    for (int c = 0; c < col_num; c++) {
      int col0 = c * 8 + (tid % 4) * 2;
      int col1 = col0 + 1;
      D(col0, row0) = cast_if<value_type>(d[c * 4]);
      D(col1, row0) = cast_if<value_type>(d[c * 4 + 1]);
      D(col0, row1) = cast_if<value_type>(d[c * 4 + 2]);
      D(col1, row1) = cast_if<value_type>(d[c * 4 + 3]);
    }
  }
};

// Store policy for 64x64x8 WGMMA (accumulator layout matches K=16)
struct Policy_WGMMA_D_M64K8 {
  template <class Tensor, typename AccumT, int N>
  __device__ static void store_mask_row(Tensor& D, AccumT* d, int row_guard) {
    int tid = threadIdx.x % 128;
    int lane = tid % 32;
    int warp = tid / 32;
    int row0 = warp * 16 + lane / 4;
    int row1 = row0 + 8;
    int col_num = N / 8;
    using value_type = typename Tensor::value_type;
    if (row0 < row_guard) {
  #pragma unroll
      for (int c = 0; c < col_num; c++) {
        int col0 = c * 8 + (tid % 4) * 2;
        int col1 = col0 + 1;
        D(row0, col0) = cast_if<value_type>(d[c * 4]);
        D(row0, col1) = cast_if<value_type>(d[c * 4 + 1]);
      }
    }
    if (row1 < row_guard) {
  #pragma unroll
      for (int c = 0; c < col_num; c++) {
        int col0 = c * 8 + (tid % 4) * 2;
        int col1 = col0 + 1;
        D(row1, col0) = cast_if<value_type>(d[c * 4 + 2]);
        D(row1, col1) = cast_if<value_type>(d[c * 4 + 3]);
      }
    }
  }

  template <class Tensor, typename AccumT, int N>
  __device__ static void store_mask_col(Tensor& D, AccumT* d, int col_guard) {
    int tid = threadIdx.x % 128;
    int lane = tid % 32;
    int warp = tid / 32;
    int row0 = warp * 16 + lane / 4;
    int row1 = row0 + 8;
    int col_num = N / 8;
    using value_type = typename Tensor::value_type;
  #pragma unroll
    for (int c = 0; c < col_num; c++) {
      int col0 = c * 8 + (tid % 4) * 2;
      int col1 = col0 + 1;
      if (col0 < col_guard) {
        D(row0, col0) = cast_if<value_type>(d[c * 4]);
        D(row1, col0) = cast_if<value_type>(d[c * 4 + 2]);
      }
      if (col1 < col_guard) {
        D(row0, col1) = cast_if<value_type>(d[c * 4 + 1]);
        D(row1, col1) = cast_if<value_type>(d[c * 4 + 3]);
      }
    }
  }

  template <class Tensor, typename AccumT, int N>
  __device__ static void store_mask_row_col(Tensor& D, AccumT* d, int row_guard,
                                            int col_guard) {
    int tid = threadIdx.x % 128;
    int lane = tid % 32;
    int warp = tid / 32;
    int row0 = warp * 16 + lane / 4;
    int row1 = row0 + 8;
    int col_num = N / 8;
    using value_type = typename Tensor::value_type;
    if (row0 < row_guard) {
  #pragma unroll
      for (int c = 0; c < col_num; c++) {
        int col0 = c * 8 + (tid % 4) * 2;
        int col1 = col0 + 1;
        if (col0 < col_guard) D(row0, col0) = cast_if<value_type>(d[c * 4]);
        if (col1 < col_guard) D(row0, col1) = cast_if<value_type>(d[c * 4 + 1]);
      }
    }
    if (row1 < row_guard) {
  #pragma unroll
      for (int c = 0; c < col_num; c++) {
        int col0 = c * 8 + (tid % 4) * 2;
        int col1 = col0 + 1;
        if (col0 < col_guard) D(row1, col0) = cast_if<value_type>(d[c * 4 + 2]);
        if (col1 < col_guard) D(row1, col1) = cast_if<value_type>(d[c * 4 + 3]);
      }
    }
  }

  template <class Tensor, typename AccumT, int N>
  __device__ static void store(Tensor& D, AccumT* d) {
    int tid = threadIdx.x % 128;
    int lane = tid % 32;             // 0-31: lane within warp
    int warp = tid / 32;             // 0-3: which warp in warp group
    int row0 = warp * 16 + lane / 4; // first row
    int row1 = row0 + 8;             // second row
    int col_num = N / 8;             // number of column pairs
    using value_type = typename Tensor::value_type;
  #pragma unroll
    for (int c = 0; c < col_num; c++) {
      int col0 = c * 8 + (tid % 4) * 2;
      int col1 = col0 + 1;
      D(row0, col0) = cast_if<value_type>(d[c * 4]);
      D(row0, col1) = cast_if<value_type>(d[c * 4 + 1]);
      D(row1, col0) = cast_if<value_type>(d[c * 4 + 2]);
      D(row1, col1) = cast_if<value_type>(d[c * 4 + 3]);
    }
  }

  template <class Tensor, typename AccumT, int N>
  __device__ static void store_trans(Tensor& D, AccumT* d) {
    int tid = threadIdx.x % 128;
    int lane = tid % 32;
    int warp = tid / 32;
    int row0 = warp * 16 + lane / 4;
    int row1 = row0 + 8;
    int col_num = N / 8;
    using value_type = typename Tensor::value_type;
  #pragma unroll
    for (int c = 0; c < col_num; c++) {
      int col0 = c * 8 + (tid % 4) * 2;
      int col1 = col0 + 1;
      D(col0, row0) = cast_if<value_type>(d[c * 4]);
      D(col1, row0) = cast_if<value_type>(d[c * 4 + 1]);
      D(col0, row1) = cast_if<value_type>(d[c * 4 + 2]);
      D(col1, row1) = cast_if<value_type>(d[c * 4 + 3]);
    }
  }
};

// Store policy for 64x64x32 WGMMA (accumulator layout matches K=16)
struct Policy_WGMMA_D_M64K32 {
  template <class Tensor, typename AccumT, int N>
  __device__ static void store_mask_col(Tensor& D, AccumT* d, int col_guard) {
    int tid = threadIdx.x % 128;
    int lane = tid % 32;
    int warp = tid / 32;
    int row0 = warp * 16 + lane / 4;
    int row1 = row0 + 8;
    int col_num = N / 8;
    using value_type = typename Tensor::value_type;
  #pragma unroll
    for (int c = 0; c < col_num; c++) {
      int col0 = c * 8 + (tid % 4) * 2;
      int col1 = col0 + 1;
      if (col0 < col_guard) {
        D(row0, col0) = cast_if<value_type>(d[c * 4]);
        D(row1, col0) = cast_if<value_type>(d[c * 4 + 2]);
      }
      if (col1 < col_guard) {
        D(row0, col1) = cast_if<value_type>(d[c * 4 + 1]);
        D(row1, col1) = cast_if<value_type>(d[c * 4 + 3]);
      }
    }
  }

  template <class Tensor, typename AccumT, int N>
  __device__ static void store_mask_row(Tensor& D, AccumT* d, int row_guard) {
    int tid = threadIdx.x % 128;
    int lane = tid % 32;
    int warp = tid / 32;
    int row0 = warp * 16 + lane / 4;
    int row1 = row0 + 8;
    int col_num = N / 8; // number of column pairs
    using value_type = typename Tensor::value_type;

    if (row0 < row_guard) {
  #pragma unroll
      for (int c = 0; c < col_num; c++) {
        int col0 = c * 8 + (tid % 4) * 2;
        int col1 = col0 + 1;
        D(row0, col0) = cast_if<value_type>(d[c * 4]);
        D(row0, col1) = cast_if<value_type>(d[c * 4 + 1]);
      }
    }

    if (row1 < row_guard) {
  #pragma unroll
      for (int c = 0; c < col_num; c++) {
        int col0 = c * 8 + (tid % 4) * 2;
        int col1 = col0 + 1;
        D(row1, col0) = cast_if<value_type>(d[c * 4 + 2]);
        D(row1, col1) = cast_if<value_type>(d[c * 4 + 3]);
      }
    }
  }

  template <class Tensor, typename AccumT, int N>
  __device__ static void store_mask_row_col(Tensor& D, AccumT* d, int row_guard,
                                            int col_guard) {
    int tid = threadIdx.x % 128;
    int lane = tid % 32;
    int warp = tid / 32;
    int row0 = warp * 16 + lane / 4;
    int row1 = row0 + 8;
    int col_num = N / 8;
    using value_type = typename Tensor::value_type;
    if (row0 < row_guard) {
  #pragma unroll
      for (int c = 0; c < col_num; c++) {
        int col0 = c * 8 + (tid % 4) * 2;
        int col1 = col0 + 1;
        if (col0 < col_guard) D(row0, col0) = cast_if<value_type>(d[c * 4]);
        if (col1 < col_guard) D(row0, col1) = cast_if<value_type>(d[c * 4 + 1]);
      }
    }
    if (row1 < row_guard) {
  #pragma unroll
      for (int c = 0; c < col_num; c++) {
        int col0 = c * 8 + (tid % 4) * 2;
        int col1 = col0 + 1;
        if (col0 < col_guard) D(row1, col0) = cast_if<value_type>(d[c * 4 + 2]);
        if (col1 < col_guard) D(row1, col1) = cast_if<value_type>(d[c * 4 + 3]);
      }
    }
  }

  template <class Tensor, typename AccumT, int N>
  __device__ static void store(Tensor& D, AccumT* d) {
    int tid = threadIdx.x % 128;
    int lane = tid % 32;
    int warp = tid / 32;
    int row0 = warp * 16 + lane / 4;
    int row1 = row0 + 8;
    int col_num = N / 8;
    using value_type = typename Tensor::value_type;
  #pragma unroll
    for (int c = 0; c < col_num; c++) {
      int col0 = c * 8 + (tid % 4) * 2;
      int col1 = col0 + 1;
      D(row0, col0) = cast_if<value_type>(d[c * 4]);
      D(row0, col1) = cast_if<value_type>(d[c * 4 + 1]);
      D(row1, col0) = cast_if<value_type>(d[c * 4 + 2]);
      D(row1, col1) = cast_if<value_type>(d[c * 4 + 3]);
    }
  }

  template <class Tensor, typename AccumT, int N>
  __device__ static void store_trans(Tensor& D, AccumT* d) {
    int tid = threadIdx.x % 128;
    int lane = tid % 32;
    int warp = tid / 32;
    int row0 = warp * 16 + lane / 4;
    int row1 = row0 + 8;
    int col_num = N / 8;
    using value_type = typename Tensor::value_type;
  #pragma unroll
    for (int c = 0; c < col_num; c++) {
      int col0 = c * 8 + (tid % 4) * 2;
      int col1 = col0 + 1;
      D(col0, row0) = cast_if<value_type>(d[c * 4]);
      D(col1, row0) = cast_if<value_type>(d[c * 4 + 1]);
      D(col0, row1) = cast_if<value_type>(d[c * 4 + 2]);
      D(col1, row1) = cast_if<value_type>(d[c * 4 + 3]);
    }
  }
};

// Store policy for 64x64x64 WGMMA (accumulator layout matches K=32 paths)
struct Policy_WGMMA_D_M64K64 {
  template <class Tensor, typename AccumT, int N>
  __device__ static void store_mask_row(Tensor& D, AccumT* d, int row_guard) {
    int tid = threadIdx.x % 128;
    int lane = tid % 32;
    int warp = tid / 32;
    int row0 = warp * 16 + lane / 4;
    int row1 = row0 + 8;
    int col_num = N / 8;
    using value_type = typename Tensor::value_type;
    if (row0 < row_guard) {
  #pragma unroll
      for (int c = 0; c < col_num; c++) {
        int col0 = c * 8 + (tid % 4) * 2;
        int col1 = col0 + 1;
        D(row0, col0) = cast_if<value_type>(d[c * 4]);
        D(row0, col1) = cast_if<value_type>(d[c * 4 + 1]);
      }
    }
    if (row1 < row_guard) {
  #pragma unroll
      for (int c = 0; c < col_num; c++) {
        int col0 = c * 8 + (tid % 4) * 2;
        int col1 = col0 + 1;
        D(row1, col0) = cast_if<value_type>(d[c * 4 + 2]);
        D(row1, col1) = cast_if<value_type>(d[c * 4 + 3]);
      }
    }
  }

  template <class Tensor, typename AccumT, int N>
  __device__ static void store_mask_col(Tensor& D, AccumT* d, int col_guard) {
    int tid = threadIdx.x % 128;
    int lane = tid % 32;
    int warp = tid / 32;
    int row0 = warp * 16 + lane / 4;
    int row1 = row0 + 8;
    int col_num = N / 8;
    using value_type = typename Tensor::value_type;
  #pragma unroll
    for (int c = 0; c < col_num; c++) {
      int col0 = c * 8 + (tid % 4) * 2;
      int col1 = col0 + 1;
      if (col0 < col_guard) {
        D(row0, col0) = cast_if<value_type>(d[c * 4]);
        D(row1, col0) = cast_if<value_type>(d[c * 4 + 2]);
      }
      if (col1 < col_guard) {
        D(row0, col1) = cast_if<value_type>(d[c * 4 + 1]);
        D(row1, col1) = cast_if<value_type>(d[c * 4 + 3]);
      }
    }
  }

  template <class Tensor, typename AccumT, int N>
  __device__ static void store_mask_row_col(Tensor& D, AccumT* d, int row_guard,
                                            int col_guard) {
    int tid = threadIdx.x % 128;
    int lane = tid % 32;
    int warp = tid / 32;
    int row0 = warp * 16 + lane / 4;
    int row1 = row0 + 8;
    int col_num = N / 8;
    using value_type = typename Tensor::value_type;
    if (row0 < row_guard) {
  #pragma unroll
      for (int c = 0; c < col_num; c++) {
        int col0 = c * 8 + (tid % 4) * 2;
        int col1 = col0 + 1;
        if (col0 < col_guard) D(row0, col0) = cast_if<value_type>(d[c * 4]);
        if (col1 < col_guard) D(row0, col1) = cast_if<value_type>(d[c * 4 + 1]);
      }
    }
    if (row1 < row_guard) {
  #pragma unroll
      for (int c = 0; c < col_num; c++) {
        int col0 = c * 8 + (tid % 4) * 2;
        int col1 = col0 + 1;
        if (col0 < col_guard) D(row1, col0) = cast_if<value_type>(d[c * 4 + 2]);
        if (col1 < col_guard) D(row1, col1) = cast_if<value_type>(d[c * 4 + 3]);
      }
    }
  }

  template <class Tensor, typename AccumT, int N>
  __device__ static void store(Tensor& D, AccumT* d) {
    int tid = threadIdx.x % 128;
    int lane = tid % 32;
    int warp = tid / 32;
    int row0 = warp * 16 + lane / 4;
    int row1 = row0 + 8;
    int col_num = N / 8;
    using value_type = typename Tensor::value_type;
  #pragma unroll
    for (int c = 0; c < col_num; c++) {
      int col0 = c * 8 + (tid % 4) * 2;
      int col1 = col0 + 1;
      D(row0, col0) = cast_if<value_type>(d[c * 4]);
      D(row0, col1) = cast_if<value_type>(d[c * 4 + 1]);
      D(row1, col0) = cast_if<value_type>(d[c * 4 + 2]);
      D(row1, col1) = cast_if<value_type>(d[c * 4 + 3]);
    }
  }

  template <class Tensor, typename AccumT, int N>
  __device__ static void store_trans(Tensor& D, AccumT* d) {
    int tid = threadIdx.x % 128;
    int lane = tid % 32;
    int warp = tid / 32;
    int row0 = warp * 16 + lane / 4;
    int row1 = row0 + 8;
    int col_num = N / 8;
    using value_type = typename Tensor::value_type;
  #pragma unroll
    for (int c = 0; c < col_num; c++) {
      int col0 = c * 8 + (tid % 4) * 2;
      int col1 = col0 + 1;
      D(col0, row0) = cast_if<value_type>(d[c * 4]);
      D(col1, row0) = cast_if<value_type>(d[c * 4 + 1]);
      D(col0, row1) = cast_if<value_type>(d[c * 4 + 2]);
      D(col1, row1) = cast_if<value_type>(d[c * 4 + 3]);
    }
  }
};

// Store policy for 64x64x256 WGMMA (binary accumulator layout matches K=16)
struct Policy_WGMMA_D_M64K256 {
  template <class Tensor, typename AccumT, int N>
  __device__ static void store_mask_row(Tensor& D, AccumT* d, int row_guard) {
    int tid = threadIdx.x % 128;
    int lane = tid % 32;
    int warp = tid / 32;
    int row0 = warp * 16 + lane / 4;
    int row1 = row0 + 8;
    int col_num = N / 8;
    using value_type = typename Tensor::value_type;
    if (row0 < row_guard) {
  #pragma unroll
      for (int c = 0; c < col_num; c++) {
        int col0 = c * 8 + (tid % 4) * 2;
        int col1 = col0 + 1;
        D(row0, col0) = cast_if<value_type>(d[c * 4]);
        D(row0, col1) = cast_if<value_type>(d[c * 4 + 1]);
      }
    }
    if (row1 < row_guard) {
  #pragma unroll
      for (int c = 0; c < col_num; c++) {
        int col0 = c * 8 + (tid % 4) * 2;
        int col1 = col0 + 1;
        D(row1, col0) = cast_if<value_type>(d[c * 4 + 2]);
        D(row1, col1) = cast_if<value_type>(d[c * 4 + 3]);
      }
    }
  }

  template <class Tensor, typename AccumT, int N>
  __device__ static void store_mask_col(Tensor& D, AccumT* d, int col_guard) {
    int tid = threadIdx.x % 128;
    int lane = tid % 32;
    int warp = tid / 32;
    int row0 = warp * 16 + lane / 4;
    int row1 = row0 + 8;
    int col_num = N / 8;
    using value_type = typename Tensor::value_type;
  #pragma unroll
    for (int c = 0; c < col_num; c++) {
      int col0 = c * 8 + (tid % 4) * 2;
      int col1 = col0 + 1;
      if (col0 < col_guard) {
        D(row0, col0) = cast_if<value_type>(d[c * 4]);
        D(row1, col0) = cast_if<value_type>(d[c * 4 + 2]);
      }
      if (col1 < col_guard) {
        D(row0, col1) = cast_if<value_type>(d[c * 4 + 1]);
        D(row1, col1) = cast_if<value_type>(d[c * 4 + 3]);
      }
    }
  }

  template <class Tensor, typename AccumT, int N>
  __device__ static void store_mask_row_col(Tensor& D, AccumT* d, int row_guard,
                                            int col_guard) {
    int tid = threadIdx.x % 128;
    int lane = tid % 32;
    int warp = tid / 32;
    int row0 = warp * 16 + lane / 4;
    int row1 = row0 + 8;
    int col_num = N / 8;
    using value_type = typename Tensor::value_type;
    if (row0 < row_guard) {
  #pragma unroll
      for (int c = 0; c < col_num; c++) {
        int col0 = c * 8 + (tid % 4) * 2;
        int col1 = col0 + 1;
        if (col0 < col_guard) D(row0, col0) = cast_if<value_type>(d[c * 4]);
        if (col1 < col_guard) D(row0, col1) = cast_if<value_type>(d[c * 4 + 1]);
      }
    }
    if (row1 < row_guard) {
  #pragma unroll
      for (int c = 0; c < col_num; c++) {
        int col0 = c * 8 + (tid % 4) * 2;
        int col1 = col0 + 1;
        if (col0 < col_guard) D(row1, col0) = cast_if<value_type>(d[c * 4 + 2]);
        if (col1 < col_guard) D(row1, col1) = cast_if<value_type>(d[c * 4 + 3]);
      }
    }
  }

  template <class Tensor, typename AccumT, int N>
  __device__ static void store(Tensor& D, AccumT* d) {
    int tid = threadIdx.x % 128;
    int lane = tid % 32;
    int warp = tid / 32;
    int row0 = warp * 16 + lane / 4;
    int row1 = row0 + 8;
    int col_num = N / 8;
    using value_type = typename Tensor::value_type;
  #pragma unroll
    for (int c = 0; c < col_num; c++) {
      int col0 = c * 8 + (tid % 4) * 2;
      int col1 = col0 + 1;
      D(row0, col0) = cast_if<value_type>(d[c * 4]);
      D(row0, col1) = cast_if<value_type>(d[c * 4 + 1]);
      D(row1, col0) = cast_if<value_type>(d[c * 4 + 2]);
      D(row1, col1) = cast_if<value_type>(d[c * 4 + 3]);
    }
  }

  template <class Tensor, typename AccumT, int N>
  __device__ static void store_trans(Tensor& D, AccumT* d) {
    int tid = threadIdx.x % 128;
    int lane = tid % 32;
    int warp = tid / 32;
    int row0 = warp * 16 + lane / 4;
    int row1 = row0 + 8;
    int col_num = N / 8;
    using value_type = typename Tensor::value_type;
  #pragma unroll
    for (int c = 0; c < col_num; c++) {
      int col0 = c * 8 + (tid % 4) * 2;
      int col1 = col0 + 1;
      D(col0, row0) = cast_if<value_type>(d[c * 4]);
      D(col1, row0) = cast_if<value_type>(d[c * 4 + 1]);
      D(col0, row1) = cast_if<value_type>(d[c * 4 + 2]);
      D(col1, row1) = cast_if<value_type>(d[c * 4 + 3]);
    }
  }
};

// unified MMA load/store fragment interfaces
// load A fragment
template <class MMA, class Tensor>
__device__ static inline auto load_fragment_a(Tensor const& A) {
  static_assert(MMA_Policy<MMA>::supported, "No policy for this MMA");
  return MMA_Policy<MMA>::typeA::load(A);
}

// load B fragment
template <class MMA, class Tensor>
__device__ static inline auto load_fragment_b(Tensor const& B) {
  static_assert(MMA_Policy<MMA>::supported, "No policy for this MMA");
  return MMA_Policy<MMA>::typeB::load(B);
}

// load E fragment (metadata)
template <class MMA, class Tensor>
__device__ static inline auto load_fragment_e(Tensor const& E) {
  static_assert(MMA_Policy<MMA>::supported, "No policy for this MMA");
  return MMA_Policy<MMA>::typeE::load(E);
}

// load d fragment with pointer
template <class MMA, int N = 0, class Tensor, class AccumT>
__device__ static inline void load_fragment_d(Tensor const& D,
                                              AccumT* const d) {
  static_assert(MMA_Policy<MMA>::supported, "No policy for this MMA");
  static_assert(std::is_same<AccumT, float>::value ||
                    std::is_same<AccumT, double>::value ||
                    std::is_same<AccumT, f16>::value ||
                    std::is_same<AccumT, s32>::value,
                "load d only supports float/double/f16/s32 accumulator type");
  static_assert(AccumTCast<typename Tensor::value_type, AccumT>::supported ||
                    std::is_same<typename Tensor::value_type, AccumT>::value,
                "load d unsupported type cast");
  if constexpr (N > 0)
    MMA_Policy<MMA>::typeD::template load<Tensor, AccumT, N>(D, d);
  else
    MMA_Policy<MMA>::typeD::template load<Tensor, AccumT>(D, d);
}

// store d fragment with pointer
template <class MMA, int N = 0, class Tensor, class AccumT>
__device__ static __forceinline__ void store_fragment_d(Tensor& D,
                                                        AccumT* const d) {
  static_assert(MMA_Policy<MMA>::supported, "No policy for this MMA");
  static_assert(std::is_same<AccumT, float>::value ||
                    std::is_same<AccumT, double>::value ||
                    std::is_same<AccumT, f16>::value ||
                    std::is_same<AccumT, s32>::value,
                "store d only supports float/double/f16/s32 accumulator type");
  static_assert(AccumTCast<typename Tensor::value_type, AccumT>::supported ||
                    std::is_same<typename Tensor::value_type, AccumT>::value,
                "store d unsupported type cast");
  if constexpr (N > 0)
    MMA_Policy<MMA>::typeD::template store<Tensor, AccumT, N>(D, d);
  else
    MMA_Policy<MMA>::typeD::template store<Tensor, AccumT>(D, d);
}

template <class MMA, int N = 0, class Tensor, class AccumT>
__device__ static __forceinline__ void
store_fragment_d_mask_row(Tensor& D, AccumT* const d, int row_guard) {
  static_assert(MMA_Policy<MMA>::supported, "No policy for this MMA");
  static_assert(std::is_same<AccumT, float>::value ||
                    std::is_same<AccumT, double>::value ||
                    std::is_same<AccumT, f16>::value ||
                    std::is_same<AccumT, s32>::value,
                "store d only supports float/double/f16/s32 accumulator type");
  static_assert(AccumTCast<typename Tensor::value_type, AccumT>::supported ||
                    std::is_same<typename Tensor::value_type, AccumT>::value,
                "store d unsupported type cast");
  if constexpr (N > 0)
    MMA_Policy<MMA>::typeD::template store_mask_row<Tensor, AccumT, N>(
        D, d, row_guard);
  else
    MMA_Policy<MMA>::typeD::template store_mask_row<Tensor, AccumT>(D, d,
                                                                    row_guard);
}

template <class MMA, int N = 0, class Tensor, class AccumT>
__device__ static inline void
store_fragment_d_mask_col(Tensor& D, AccumT* const d, int col_guard) {
  static_assert(MMA_Policy<MMA>::supported, "No policy for this MMA");
  static_assert(std::is_same<AccumT, float>::value ||
                    std::is_same<AccumT, double>::value ||
                    std::is_same<AccumT, f16>::value ||
                    std::is_same<AccumT, s32>::value,
                "store d only supports float/double/f16/s32 accumulator type");
  static_assert(AccumTCast<typename Tensor::value_type, AccumT>::supported ||
                    std::is_same<typename Tensor::value_type, AccumT>::value,
                "store d unsupported type cast");
  if constexpr (N > 0)
    MMA_Policy<MMA>::typeD::template store_mask_col<Tensor, AccumT, N>(
        D, d, col_guard);
  else
    MMA_Policy<MMA>::typeD::template store_mask_col<Tensor, AccumT>(D, d,
                                                                    col_guard);
}

template <class MMA, int N = 0, class Tensor, class AccumT>
__device__ static inline void
store_fragment_d_mask_row_col(Tensor& D, AccumT* const d, int row_guard,
                              int col_guard) {
  static_assert(MMA_Policy<MMA>::supported, "No policy for this MMA");
  static_assert(std::is_same<AccumT, float>::value ||
                    std::is_same<AccumT, double>::value ||
                    std::is_same<AccumT, f16>::value ||
                    std::is_same<AccumT, s32>::value,
                "store d only supports float/double/f16/s32 accumulator type");
  static_assert(AccumTCast<typename Tensor::value_type, AccumT>::supported ||
                    std::is_same<typename Tensor::value_type, AccumT>::value,
                "store d unsupported type cast");
  if constexpr (N > 0)
    MMA_Policy<MMA>::typeD::template store_mask_row_col<Tensor, AccumT, N>(
        D, d, row_guard, col_guard);
  else
    MMA_Policy<MMA>::typeD::template store_mask_row_col<Tensor, AccumT>(
        D, d, row_guard, col_guard);
}

template <class MMA, int N = 0, class Tensor, class AccumT>
__device__ static inline void store_fragment_d_trans(Tensor& D,
                                                     AccumT* const d) {
  static_assert(MMA_Policy<MMA>::supported, "No policy for this MMA");
  static_assert(std::is_same<AccumT, float>::value ||
                    std::is_same<AccumT, double>::value ||
                    std::is_same<AccumT, f16>::value ||
                    std::is_same<AccumT, s32>::value,
                "store d only supports float/double/f16/s32 accumulator type");
  static_assert(AccumTCast<typename Tensor::value_type, AccumT>::supported ||
                    std::is_same<typename Tensor::value_type, AccumT>::value,
                "store d unsupported type cast");
  if constexpr (N > 0)
    MMA_Policy<MMA>::typeD::template store_trans<Tensor, AccumT, N>(D, d);
  else
    MMA_Policy<MMA>::typeD::template store_trans<Tensor, AccumT>(D, d);
}

template <class MMA, int N, class Tensor>
__device__ static inline void
store_fragment_d_stmatrix_f32_bf16(Tensor& D, float* const d) {
  static_assert(std::is_same<typename Tensor::value_type, bf16>::value,
                "stmatrix f32->bf16 store requires bf16 output tensor");
  static_assert(
      (N % 8) == 0,
      "stmatrix f32->bf16 store currently requires N to be divisible by 8");

  int tid = threadIdx.x % 128;
  int lane = tid % 32;
  int warp = tid / 32;

  bf16* d_base_ptr = &D(0, 0);
  uint32_t d_base_addr =
      static_cast<uint32_t>(__cvta_generic_to_shared(d_base_ptr));

  constexpr int w_iters = N / 16;
  constexpr int tail_cols = N % 16;
  constexpr int full_cols = w_iters * 16;
  constexpr uint32_t kStepBytes = static_cast<uint32_t>(16 * sizeof(bf16));

  uint32_t lane_offset =
      static_cast<uint32_t>((lane % 8) * N + (lane / 16) * N * 8 + (lane & 8));
  uint32_t addr = d_base_addr +
                  static_cast<uint32_t>((warp * 16 * N) * sizeof(bf16)) +
                  lane_offset * sizeof(bf16);

  #pragma unroll
  for (int w = 0; w < w_iters; ++w) {
    int base = w * 8;
    bf16 v0 = utils::from_f32<bf16>(d[base + 0]);
    bf16 v1 = utils::from_f32<bf16>(d[base + 1]);
    bf16 v2 = utils::from_f32<bf16>(d[base + 2]);
    bf16 v3 = utils::from_f32<bf16>(d[base + 3]);
    bf16 v4 = utils::from_f32<bf16>(d[base + 4]);
    bf16 v5 = utils::from_f32<bf16>(d[base + 5]);
    bf16 v6 = utils::from_f32<bf16>(d[base + 6]);
    bf16 v7 = utils::from_f32<bf16>(d[base + 7]);

    uint32_t r0 = static_cast<uint32_t>(bitcast_uint(v0)) |
                  (static_cast<uint32_t>(bitcast_uint(v1)) << 16);
    uint32_t r1 = static_cast<uint32_t>(bitcast_uint(v4)) |
                  (static_cast<uint32_t>(bitcast_uint(v5)) << 16);
    uint32_t r2 = static_cast<uint32_t>(bitcast_uint(v2)) |
                  (static_cast<uint32_t>(bitcast_uint(v3)) << 16);
    uint32_t r3 = static_cast<uint32_t>(bitcast_uint(v6)) |
                  (static_cast<uint32_t>(bitcast_uint(v7)) << 16);

    asm volatile("stmatrix.sync.aligned.m8n8.x4.shared::cta.b16 [%0], "
                 "{%1, %2, %3, %4};\n"
                 :
                 : "r"(addr), "r"(r0), "r"(r1), "r"(r2), "r"(r3));
    addr += kStepBytes;
  }

  if constexpr (tail_cols != 0) {
    static_assert(
        tail_cols == 8,
        "stmatrix f32->bf16 tail path only supports one 8-column remainder");
    int row0 = warp * 16 + lane / 4;
    int row1 = row0 + 8;
    int col0 = full_cols + (tid % 4) * 2;
    int col1 = col0 + 1;
    int base = w_iters * 8;

    D(row0, col0) = utils::from_f32<bf16>(d[base + 0]);
    D(row0, col1) = utils::from_f32<bf16>(d[base + 1]);
    D(row1, col0) = utils::from_f32<bf16>(d[base + 2]);
    D(row1, col1) = utils::from_f32<bf16>(d[base + 3]);
  }
}

template <class MMA, int N, class Tensor>
__device__ static inline void
store_fragment_d_stmatrix_f32_f16(Tensor& D, float* const d) {
  static_assert(std::is_same<typename Tensor::value_type, f16>::value,
                "stmatrix f32->f16 store requires f16 output tensor");
  static_assert(
      (N % 8) == 0,
      "stmatrix f32->f16 store currently requires N to be divisible by 8");

  int tid = threadIdx.x % 128;
  int lane = tid % 32;
  int warp = tid / 32;

  f16* d_base_ptr = &D(0, 0);
  uint32_t d_base_addr =
      static_cast<uint32_t>(__cvta_generic_to_shared(d_base_ptr));

  constexpr int w_iters = N / 16;
  constexpr int tail_cols = N % 16;
  constexpr int full_cols = w_iters * 16;
  constexpr uint32_t kStepBytes = static_cast<uint32_t>(16 * sizeof(f16));

  uint32_t lane_offset =
      static_cast<uint32_t>((lane % 8) * N + (lane / 16) * N * 8 + (lane & 8));
  uint32_t addr = d_base_addr +
                  static_cast<uint32_t>((warp * 16 * N) * sizeof(f16)) +
                  lane_offset * sizeof(f16);

  #pragma unroll
  for (int w = 0; w < w_iters; ++w) {
    int base = w * 8;
    f16 v0 = utils::from_f32<f16>(d[base + 0]);
    f16 v1 = utils::from_f32<f16>(d[base + 1]);
    f16 v2 = utils::from_f32<f16>(d[base + 2]);
    f16 v3 = utils::from_f32<f16>(d[base + 3]);
    f16 v4 = utils::from_f32<f16>(d[base + 4]);
    f16 v5 = utils::from_f32<f16>(d[base + 5]);
    f16 v6 = utils::from_f32<f16>(d[base + 6]);
    f16 v7 = utils::from_f32<f16>(d[base + 7]);

    uint32_t r0 = static_cast<uint32_t>(bitcast_uint(v0)) |
                  (static_cast<uint32_t>(bitcast_uint(v1)) << 16);
    uint32_t r1 = static_cast<uint32_t>(bitcast_uint(v4)) |
                  (static_cast<uint32_t>(bitcast_uint(v5)) << 16);
    uint32_t r2 = static_cast<uint32_t>(bitcast_uint(v2)) |
                  (static_cast<uint32_t>(bitcast_uint(v3)) << 16);
    uint32_t r3 = static_cast<uint32_t>(bitcast_uint(v6)) |
                  (static_cast<uint32_t>(bitcast_uint(v7)) << 16);

    asm volatile("stmatrix.sync.aligned.m8n8.x4.shared::cta.b16 [%0], "
                 "{%1, %2, %3, %4};\n"
                 :
                 : "r"(addr), "r"(r0), "r"(r1), "r"(r2), "r"(r3));
    addr += kStepBytes;
  }

  if constexpr (tail_cols != 0) {
    static_assert(
        tail_cols == 8,
        "stmatrix f32->f16 tail path only supports one 8-column remainder");
    int row0 = warp * 16 + lane / 4;
    int row1 = row0 + 8;
    int col0 = full_cols + (tid % 4) * 2;
    int col1 = col0 + 1;
    int base = w_iters * 8;

    D(row0, col0) = utils::from_f32<f16>(d[base + 0]);
    D(row0, col1) = utils::from_f32<f16>(d[base + 1]);
    D(row1, col0) = utils::from_f32<f16>(d[base + 2]);
    D(row1, col1) = utils::from_f32<f16>(d[base + 3]);
  }
}

template <class MMA, int N, class Tensor>
__device__ static inline void
store_fragment_d_stmatrix_trans_f32_bf16(Tensor& D, float* const d) {
  static_assert(
      std::is_same<typename Tensor::value_type, bf16>::value,
      "transposed stmatrix f32->bf16 store requires bf16 output tensor");
  static_assert((N % 8) == 0, "transposed stmatrix f32->bf16 store currently "
                              "requires N to be divisible by 8");

  int tid = threadIdx.x % 128;
  int lane = tid % 32;
  int warp = tid / 32;

  bf16* d_base_ptr = &D(0, 0);
  uint32_t d_base_addr =
      static_cast<uint32_t>(__cvta_generic_to_shared(d_base_ptr));

  const uint32_t ld = static_cast<uint32_t>(&D(1, 0) - &D(0, 0));
  constexpr int w_iters = N / 16;
  const uint32_t kStepBytes = static_cast<uint32_t>(16 * ld * sizeof(bf16));

  uint32_t lane_offset = static_cast<uint32_t>(warp * 16) +
                         static_cast<uint32_t>(lane % 8) * ld +
                         static_cast<uint32_t>(lane / 16) * ld * 8 +
                         static_cast<uint32_t>(lane & 8);
  uint32_t addr = d_base_addr + lane_offset * sizeof(bf16);

  #pragma unroll
  for (int w = 0; w < w_iters; ++w) {
    int base = w * 8;
    bf16 v0 = utils::from_f32<bf16>(d[base + 0]);
    bf16 v1 = utils::from_f32<bf16>(d[base + 1]);
    bf16 v2 = utils::from_f32<bf16>(d[base + 2]);
    bf16 v3 = utils::from_f32<bf16>(d[base + 3]);
    bf16 v4 = utils::from_f32<bf16>(d[base + 4]);
    bf16 v5 = utils::from_f32<bf16>(d[base + 5]);
    bf16 v6 = utils::from_f32<bf16>(d[base + 6]);
    bf16 v7 = utils::from_f32<bf16>(d[base + 7]);

    uint32_t r0 = static_cast<uint32_t>(bitcast_uint(v0)) |
                  (static_cast<uint32_t>(bitcast_uint(v1)) << 16);
    uint32_t r1 = static_cast<uint32_t>(bitcast_uint(v2)) |
                  (static_cast<uint32_t>(bitcast_uint(v3)) << 16);
    uint32_t r2 = static_cast<uint32_t>(bitcast_uint(v4)) |
                  (static_cast<uint32_t>(bitcast_uint(v5)) << 16);
    uint32_t r3 = static_cast<uint32_t>(bitcast_uint(v6)) |
                  (static_cast<uint32_t>(bitcast_uint(v7)) << 16);

    asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared::cta.b16 [%0], "
                 "{%1, %2, %3, %4};\n"
                 :
                 : "r"(addr), "r"(r0), "r"(r1), "r"(r2), "r"(r3));
    addr += kStepBytes;
  }
}

// stmatrix-based store for WGMMA accumulators (--stmatrix flag).
//
// Uses PTX stmatrix.sync.aligned.m8n8.x1.b16 to write 8x8 sub-tiles of
// the WGMMA accumulator to shared memory in a bank-conflict-free manner.
//
// Because stmatrix stores in dense row-major stride-8 format (128-byte
// blocks), but the destination tensor D has stride N, we use a per-warp
// temp buffer in shared memory: stmatrix writes to the temp buffer, then
// threads scatter to the correct strided locations in D.
//
// A register shuffle (__shfl_sync) re-maps the WGMMA accumulator register
// layout (row=lane/4, col_pair=lane%4) to the stmatrix-expected layout
// (row=lane%8, col_pair=lane/8) before each stmatrix call.
//
// NOTE: This is a demonstrative implementation.  For maximum performance,
//       the output shared memory layout should be changed to tile-8 format
//       so that stmatrix can write directly without the scatter step, and
//       TMA output should use a matching swizzled tensor map.
template <class MMA, int N, class Tensor, class AccumT>
__device__ static inline void store_fragment_d_stmatrix(Tensor& D,
                                                        AccumT* const d) {
  static_assert(sizeof(AccumT) == 2,
                "stmatrix store requires 16-bit accumulator type, f16 or bf16");
  static_assert((N % 8) == 0,
                "stmatrix store currently requires N to be divisible by 8");

  int tid = threadIdx.x % 128;
  int lane = tid % 32;
  int warp = tid / 32;

  AccumT* d_base_ptr = &D(0, 0);
  uint32_t d_base_addr =
      static_cast<uint32_t>(__cvta_generic_to_shared(d_base_ptr));

  constexpr int w_iters = N / 16;
  constexpr uint32_t kStepBytes = static_cast<uint32_t>(16 * sizeof(AccumT));

  uint32_t lane_offset =
      static_cast<uint32_t>((lane % 8) * N + (lane / 16) * N * 8 + (lane & 8));
  uint32_t addr = d_base_addr +
                  static_cast<uint32_t>((warp * 16 * N) * sizeof(AccumT)) +
                  lane_offset * sizeof(AccumT);
  auto d_u32 = reinterpret_cast<uint32_t const*>(d);

  #pragma unroll
  for (int w = 0; w < w_iters; ++w) {
    uint32_t r0 = d_u32[w * 4 + 0];
    uint32_t r1 = d_u32[w * 4 + 2];
    uint32_t r2 = d_u32[w * 4 + 1];
    uint32_t r3 = d_u32[w * 4 + 3];

    asm volatile("stmatrix.sync.aligned.m8n8.x4.shared::cta.b16 [%0], "
                 "{%1, %2, %3, %4};\n"
                 :
                 : "r"(addr), "r"(r0), "r"(r1), "r"(r2), "r"(r3));
    addr += kStepBytes;
  }
}

template <class MMA, int N, class Tensor, class AccumT>
__device__ static inline void store_fragment_d_stmatrix_trans(Tensor& D,
                                                              AccumT* const d) {
  static_assert(sizeof(AccumT) == 2,
                "stmatrix store requires 16-bit accumulator type, f16 or bf16");
  static_assert((N % 8) == 0,
                "stmatrix store currently requires N to be divisible by 8");

  int tid = threadIdx.x % 128;
  int lane = tid % 32;
  int warp = tid / 32;

  AccumT* d_base_ptr = &D(0, 0);
  uint32_t d_base_addr =
      static_cast<uint32_t>(__cvta_generic_to_shared(d_base_ptr));

  const uint32_t ld = static_cast<uint32_t>(&D(1, 0) - &D(0, 0));
  constexpr int w_iters = N / 16;
  const uint32_t kStepBytes = static_cast<uint32_t>(16 * ld * sizeof(AccumT));

  uint32_t lane_offset = static_cast<uint32_t>(warp * 16) +
                         static_cast<uint32_t>(lane % 8) * ld +
                         static_cast<uint32_t>(lane / 16) * ld * 8 +
                         static_cast<uint32_t>(lane & 8);
  uint32_t addr = d_base_addr + lane_offset * sizeof(AccumT);
  auto d_u32 = reinterpret_cast<uint32_t const*>(d);

  #pragma unroll
  for (int w = 0; w < w_iters; ++w) {
    uint32_t r0 = d_u32[w * 4 + 0];
    uint32_t r1 = d_u32[w * 4 + 1];
    uint32_t r2 = d_u32[w * 4 + 2];
    uint32_t r3 = d_u32[w * 4 + 3];

    asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared::cta.b16 [%0], "
                 "{%1, %2, %3, %4};\n"
                 :
                 : "r"(addr), "r"(r0), "r"(r1), "r"(r2), "r"(r3));
    addr += kStepBytes;
  }
}

// M64 WGMMA accumulator scaling.
//
// d[i] += scale_d[i] * scale_a[row] * scale_b
//
// Thread->row mapping (M64 WGMMA layout, warpgroup of 128 threads):
//   row0 = warp*16 + (lane>>2)       in [0..55]
//   row1 = row0 + 8                  in [8..63]
//
// valid_rows: number of valid rows in scale_a. Threads whose rows
// exceed this are zeroed. When valid_rows >= 64 (the full M64 tile),
// the masking comparisons always pass and the cost is negligible.
// A single implementation avoids duplicate template instantiations
// that would increase register pressure in the calling kernel.
template <typename AccT, typename ScaleT, int N>
__device__ __forceinline__ void
scale_accumulator(AccT* d, AccT* scale_d, ScaleT* scale_a_ptr, int scale_a_ld,
                  int valid_rows, ScaleT scale_b) {
  static_assert(std::is_same_v<ScaleT, f32>,
                "scale_accumulator: ScaleT must be f32");
  static_assert(std::is_same_v<AccT, f16> || std::is_same_v<AccT, float>,
                "scale_accumulator: AccT must be f16 or float");

  int itd = threadIdx.x & 127;
  int lane = itd & 31;
  int warp = itd >> 5;
  int row0 = warp * 16 + (lane >> 2);
  int row1 = row0 + 8;
  constexpr int col_num = N / 8;

  int row0_valid = row0 < valid_rows;
  int row1_valid = row1 < valid_rows;
  auto* sa_ptr0 = scale_a_ptr + (row0_valid ? row0 : 0) * scale_a_ld;
  auto* sa_ptr1 = scale_a_ptr + (row1_valid ? row1 : 0) * scale_a_ld;

  float sa0 = 0.0f, sa1 = 0.0f;
  #if defined(__CUDA_ARCH__)
  if (__isShared(scale_a_ptr)) {
    sa0 = *sa_ptr0;
    sa1 = *sa_ptr1;
  } else if (__isGlobal(scale_a_ptr)) {
    sa0 = __ldg(sa_ptr0);
    sa1 = __ldg(sa_ptr1);
  } else {
    sa0 = *sa_ptr0;
    sa1 = *sa_ptr1;
  }
  #else
  sa0 = *sa_ptr0;
  sa1 = *sa_ptr1;
  #endif
  sa0 *= scale_b * static_cast<float>(row0_valid);
  sa1 *= scale_b * static_cast<float>(row1_valid);

  if constexpr (std::is_same_v<AccT, f16>) {
  #if defined(__USE_CUDA_TYPE__)
    auto* d2 = reinterpret_cast<__half2*>(d);
    auto const* scale_d2 = reinterpret_cast<__half2 const*>(scale_d);
    __half2 sa0_h2 = __float2half2_rn(sa0);
    __half2 sa1_h2 = __float2half2_rn(sa1);
    #pragma unroll
    for (int c = 0; c < col_num; c++) {
      int base2 = c * 2;
      d2[base2 + 0] = __hfma2(scale_d2[base2 + 0], sa0_h2, d2[base2 + 0]);
      d2[base2 + 1] = __hfma2(scale_d2[base2 + 1], sa1_h2, d2[base2 + 1]);
    }
  #else
    #pragma unroll
    for (int c = 0; c < col_num; c++) {
      int base = c * 4;
      d[base + 0] += utils::from_f32<f16>(to_f32(scale_d[base + 0]) * sa0);
      d[base + 1] += utils::from_f32<f16>(to_f32(scale_d[base + 1]) * sa0);
      d[base + 2] += utils::from_f32<f16>(to_f32(scale_d[base + 2]) * sa1);
      d[base + 3] += utils::from_f32<f16>(to_f32(scale_d[base + 3]) * sa1);
    }
  #endif
  } else if constexpr (std::is_same_v<AccT, float>) {
  #pragma unroll
    for (int c = 0; c < col_num; c++) {
      int base = c * 4;
      d[base + 0] = fmaf(scale_d[base + 0], sa0, d[base + 0]);
      d[base + 1] = fmaf(scale_d[base + 1], sa0, d[base + 1]);
      d[base + 2] = fmaf(scale_d[base + 2], sa1, d[base + 2]);
      d[base + 3] = fmaf(scale_d[base + 3], sa1, d[base + 3]);
    }
  }
}

// --------------- MMA policy specializations ---------------
struct MMA {};
struct CUTE_MMA : MMA {};
struct CUTE_MMA_M8N8K4 : CUTE_MMA {};
struct CUTE_MMA_M8N8K16 : CUTE_MMA {};
struct CUTE_MMA_M16N8K4 : CUTE_MMA {};
struct CUTE_MMA_M16N8K8 : CUTE_MMA {};
struct CUTE_MMA_M16N8K16 : CUTE_MMA {};
struct CUTE_MMA_M16N8K32 : CUTE_MMA {};
struct CUTE_MMA_M16N8K128 : CUTE_MMA {};
// Sparse MMA atom types for 2:4 structured sparsity
struct CUTE_MMA_SPARSE_M16N8K16 : CUTE_MMA {};
struct CUTE_MMA_SPARSE_M16N8K32 : CUTE_MMA {};
struct CUTE_MMA_SPARSE_M16N8K64 : CUTE_MMA {};
struct CUTE_WGMMA : MMA {};
struct CUTE_WGMMA_M64K8 : CUTE_WGMMA {};
struct CUTE_WGMMA_M64K16 : CUTE_WGMMA {};
struct CUTE_WGMMA_M64K32 : CUTE_WGMMA {};
struct CUTE_WGMMA_M64K64 : CUTE_WGMMA {};
struct CUTE_WGMMA_M64k256 : CUTE_WGMMA {};

template <>
struct MMA_Policy<CUTE_MMA_M8N8K4> {
  static constexpr bool supported = true;
  using typeA = Policy_A_M8N8K4;
  using typeB = Policy_B_M8N8K4;
  using typeD = Policy_D_M8N8;
};

template <>
struct MMA_Policy<CUTE_MMA_M8N8K16> {
  static constexpr bool supported = true;
  using typeA = Policy_A_M8N8K16;
  using typeB = Policy_B_M8N8K16;
  using typeD = Policy_D_M8N8;
};

template <>
struct MMA_Policy<CUTE_MMA_M16N8K4> {
  static constexpr bool supported = true;
  using typeA = Policy_A_M16N8K4;
  using typeB = Policy_B_M16N8K4;
  using typeD = Policy_D_M16N8;
};

template <>
struct MMA_Policy<CUTE_MMA_M16N8K8> {
  static constexpr bool supported = true;
  using typeA = Policy_A_M16N8K8;
  using typeB = Policy_B_M16N8K8;
  using typeD = Policy_D_M16N8;
};

template <>
struct MMA_Policy<CUTE_MMA_M16N8K16> {
  static constexpr bool supported = true;
  using typeA = Policy_A_M16N8K16;
  using typeB = Policy_B_M16N8K16;
  using typeD = Policy_D_M16N8;
};

template <>
struct MMA_Policy<CUTE_MMA_M16N8K32> {
  static constexpr bool supported = true;
  using typeA = Policy_A_M16N8K32;
  using typeB = Policy_B_M16N8K32;
  using typeD = Policy_D_M16N8;
};

// Sparse MMA policies for 2:4 structured sparsity
template <>
struct MMA_Policy<CUTE_MMA_SPARSE_M16N8K16> {
  static constexpr bool supported = true;
  using typeA = Policy_A_Sparse_M16N8K16;
  using typeB = Policy_B_Sparse_M16N8K16;
  using typeD = Policy_D_M16N8;
  using typeE = Policy_E_Sparse_M16N8K16;
};

template <>
struct MMA_Policy<CUTE_MMA_SPARSE_M16N8K32> {
  static constexpr bool supported = true;
  using typeA = Policy_A_Sparse_M16N8K32;
  using typeB = Policy_B_Sparse_M16N8K32;
  using typeD = Policy_D_M16N8;
  using typeE = Policy_E_Sparse_M16N8K32;
};

template <>
struct MMA_Policy<CUTE_MMA_SPARSE_M16N8K64> {
  static constexpr bool supported = true;
  using typeA = Policy_A_Sparse_M16N8K64;
  using typeB = Policy_B_Sparse_M16N8K64;
  using typeD = Policy_D_M16N8;
  using typeE = Policy_E_Sparse_M16N8K64;
};

// wgmma policies
template <>
struct MMA_Policy<CUTE_WGMMA_M64K16> {

  static constexpr bool supported = true;
  using typeD = Policy_WGMMA_D_M64K16;
};

template <>
struct MMA_Policy<CUTE_WGMMA_M64K8> {
  static constexpr bool supported = true;
  using typeD = Policy_WGMMA_D_M64K8;
};

template <>
struct MMA_Policy<CUTE_WGMMA_M64K32> {
  static constexpr bool supported = true;
  using typeD = Policy_WGMMA_D_M64K32;
};

template <>
struct MMA_Policy<CUTE_WGMMA_M64K64> {
  static constexpr bool supported = true;
  using typeD = Policy_WGMMA_D_M64K64;
};

template <>
struct MMA_Policy<CUTE_WGMMA_M64k256> {
  static constexpr bool supported = true;
  using typeD = Policy_WGMMA_D_M64K256;
};

// TODO: all 16x8x64 (sub byte)

// TODO: all 16x8x128 (b1)

// TODO: all 16x8x256 (b1)

// --------------- WGMMA policies (SM90+) ---------------
// Note: WGMMA uses PTX inline assembly directly via wgmma_m64n64k16<>
// template. No MMA_Policy specializations are needed for WGMMA as it bypasses
// the cute MMA policy system.

} // namespace choreo

namespace cute {

// --------------- Sparse MMA PTX wrappers for SM80+ ---------------
// These implement mma.sp.sync.aligned instructions for 2:4 structured sparsity

// fp16 m16n8k16 sparse MMA: C = A_sparse * B + C
struct SM80_SPARSE_16x8x16_F32F16F16F32_TN {
  using DRegisters = float[4];
  using ARegisters = uint32_t[2];
  using BRegisters = uint32_t[2];
  using CRegisters = float[4];

  CUTE_HOST_DEVICE static void fma(float& d0, float& d1, float& d2, float& d3,
                                   uint32_t const& a0, uint32_t const& a1,
                                   uint32_t const& b0, uint32_t const& b1,
                                   float const& c0, float const& c1,
                                   float const& c2, float const& c3,
                                   uint32_t const& e, int const& spsel = 0) {
  #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    #if (__CUDACC_VER_MAJOR__ > 12) ||                                         \
        (__CUDACC_VER_MAJOR__ == 12 && __CUDACC_VER_MINOR__ >= 3)
    asm volatile(
        "mma.sp::ordered_metadata.sync.aligned.m16n8k16.row.col.f32.f16.f16."
        "f32 "
        "{%0, %1, %2, %3}, {%4, %5}, {%6, %7}, {%0, %1, %2, %3}, %8, 0x0;\n"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
        : "r"(a0), "r"(a1), "r"(b0), "r"(b1), "r"(e));
    #else
    asm volatile(
        "mma.sp.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0, %1, %2, %3}, {%4, %5}, {%6, %7}, {%0, %1, %2, %3}, %8, 0x0;\n"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
        : "r"(a0), "r"(a1), "r"(b0), "r"(b1), "r"(e));
    #endif
  #endif
  }
};

// fp16 m16n8k32 sparse MMA: C = A_sparse * B + C
struct SM80_SPARSE_16x8x32_F32F16F16F32_TN {
  using DRegisters = float[4];
  using ARegisters = uint32_t[4];
  using BRegisters = uint32_t[4];
  using CRegisters = float[4];

  CUTE_HOST_DEVICE static void
  fma(float& d0, float& d1, float& d2, float& d3, uint32_t const& a0,
      uint32_t const& a1, uint32_t const& a2, uint32_t const& a3,
      uint32_t const& b0, uint32_t const& b1, uint32_t const& b2,
      uint32_t const& b3, float const& c0, float const& c1, float const& c2,
      float const& c3, uint32_t const& e, int const& spsel = 0) {
  #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    #if (__CUDACC_VER_MAJOR__ > 12) ||                                         \
        (__CUDACC_VER_MAJOR__ == 12 && __CUDACC_VER_MINOR__ >= 3)
    asm volatile("mma.sp::ordered_metadata.sync.aligned.m16n8k32.row.col.f32."
                 "f16.f16.f32 "
                 "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9, %10, %11}, {%0, "
                 "%1, %2, %3}, %12, 0x0;\n"
                 : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1),
                   "r"(b2), "r"(b3), "r"(e));
    #else
    asm volatile("mma.sp.sync.aligned.m16n8k32.row.col.f32.f16.f16.f32 "
                 "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9, %10, %11}, {%0, "
                 "%1, %2, %3}, %12, 0x0;\n"
                 : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1),
                   "r"(b2), "r"(b3), "r"(e));
    #endif
  #endif
  }
};

// bf16 m16n8k16 sparse MMA: C = A_sparse * B + C
struct SM80_SPARSE_16x8x16_F32BF16BF16F32_TN {
  using DRegisters = float[4];
  using ARegisters = uint32_t[2];
  using BRegisters = uint32_t[2];
  using CRegisters = float[4];

  CUTE_HOST_DEVICE static void fma(float& d0, float& d1, float& d2, float& d3,
                                   uint32_t const& a0, uint32_t const& a1,
                                   uint32_t const& b0, uint32_t const& b1,
                                   float const& c0, float const& c1,
                                   float const& c2, float const& c3,
                                   uint32_t const& e, int const& spsel = 0) {
  #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    #if (__CUDACC_VER_MAJOR__ > 12) ||                                         \
        (__CUDACC_VER_MAJOR__ == 12 && __CUDACC_VER_MINOR__ >= 3)
    asm volatile(
        "mma.sp::ordered_metadata.sync.aligned.m16n8k16.row.col.f32.bf16.bf16."
        "f32 "
        "{%0, %1, %2, %3}, {%4, %5}, {%6, %7}, {%0, %1, %2, %3}, %8, 0x0;\n"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
        : "r"(a0), "r"(a1), "r"(b0), "r"(b1), "r"(e));
    #else
    asm volatile(
        "mma.sp.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0, %1, %2, %3}, {%4, %5}, {%6, %7}, {%0, %1, %2, %3}, %8, 0x0;\n"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
        : "r"(a0), "r"(a1), "r"(b0), "r"(b1), "r"(e));
    #endif
  #endif
  }
};

// bf16 m16n8k32 sparse MMA: C = A_sparse * B + C
struct SM80_SPARSE_16x8x32_F32BF16BF16F32_TN {
  using DRegisters = float[4];
  using ARegisters = uint32_t[4];
  using BRegisters = uint32_t[4];
  using CRegisters = float[4];

  CUTE_HOST_DEVICE static void
  fma(float& d0, float& d1, float& d2, float& d3, uint32_t const& a0,
      uint32_t const& a1, uint32_t const& a2, uint32_t const& a3,
      uint32_t const& b0, uint32_t const& b1, uint32_t const& b2,
      uint32_t const& b3, float const& c0, float const& c1, float const& c2,
      float const& c3, uint32_t const& e, int const& spsel = 0) {
  #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    #if (__CUDACC_VER_MAJOR__ > 12) ||                                         \
        (__CUDACC_VER_MAJOR__ == 12 && __CUDACC_VER_MINOR__ >= 3)
    asm volatile("mma.sp::ordered_metadata.sync.aligned.m16n8k32.row.col.f32."
                 "bf16.bf16.f32 "
                 "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9, %10, %11}, {%0, "
                 "%1, %2, %3}, %12, 0x0;\n"
                 : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1),
                   "r"(b2), "r"(b3), "r"(e));
    #else
    asm volatile("mma.sp.sync.aligned.m16n8k32.row.col.f32.bf16.bf16.f32 "
                 "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9, %10, %11}, {%0, "
                 "%1, %2, %3}, %12, 0x0;\n"
                 : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1),
                   "r"(b2), "r"(b3), "r"(e));
    #endif
  #endif
  }
};

  #ifdef __CHOREO_TARGET_NATIVE_FP8_SUPPORT__
// fp8 m16n8k64 sparse MMA: C = A_sparse * B + C (SM90+)
struct SM90_SPARSE_16x8x64_F16E4M3E4M3F16_TN {
  using DRegisters = uint32_t[2];
  using ARegisters = uint32_t[4];
  using BRegisters = uint32_t[4];
  using CRegisters = uint32_t[2];

  CUTE_HOST_DEVICE static void fma(uint32_t& d0, uint32_t& d1,
                                   uint32_t const& a0, uint32_t const& a1,
                                   uint32_t const& a2, uint32_t const& a3,
                                   uint32_t const& b0, uint32_t const& b1,
                                   uint32_t const& b2, uint32_t const& b3,
                                   uint32_t const& c0, uint32_t const& c1,
                                   uint32_t const& e, int const& spsel = 0) {
    #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 890
    (void)spsel;
      #if (__CUDACC_VER_MAJOR__ > 12) ||                                       \
          (__CUDACC_VER_MAJOR__ == 12 && __CUDACC_VER_MINOR__ >= 3)
    asm volatile(
        "mma.sp::ordered_metadata.sync.aligned.m16n8k64.row.col.f16.e4m3.e4m3."
        "f16 "
        "{%0, %1}, {%2, %3, %4, %5}, {%6, %7, %8, %9}, {%10, %11}, %12, 0x0;\n"
        : "=r"(d0), "=r"(d1)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "r"(b2),
          "r"(b3), "r"(c0), "r"(c1), "r"(e));
      #else
    asm volatile(
        "mma.sp.sync.aligned.m16n8k64.row.col.f16.e4m3.e4m3.f16 "
        "{%0, %1}, {%2, %3, %4, %5}, {%6, %7, %8, %9}, {%10, %11}, %12, 0x0;\n"
        : "=r"(d0), "=r"(d1)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "r"(b2),
          "r"(b3), "r"(c0), "r"(c1), "r"(e));
      #endif
    #endif
  }
};

struct SM90_SPARSE_16x8x64_F16E4M3E5M2F16_TN {
  using DRegisters = uint32_t[2];
  using ARegisters = uint32_t[4];
  using BRegisters = uint32_t[4];
  using CRegisters = uint32_t[2];

  CUTE_HOST_DEVICE static void fma(uint32_t& d0, uint32_t& d1,
                                   uint32_t const& a0, uint32_t const& a1,
                                   uint32_t const& a2, uint32_t const& a3,
                                   uint32_t const& b0, uint32_t const& b1,
                                   uint32_t const& b2, uint32_t const& b3,
                                   uint32_t const& c0, uint32_t const& c1,
                                   uint32_t const& e, int const& spsel = 0) {
    #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 890
    (void)spsel;
      #if (__CUDACC_VER_MAJOR__ > 12) ||                                       \
          (__CUDACC_VER_MAJOR__ == 12 && __CUDACC_VER_MINOR__ >= 3)
    asm volatile(
        "mma.sp::ordered_metadata.sync.aligned.m16n8k64.row.col.f16.e4m3.e5m2."
        "f16 "
        "{%0, %1}, {%2, %3, %4, %5}, {%6, %7, %8, %9}, {%10, %11}, %12, 0x0;\n"
        : "=r"(d0), "=r"(d1)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "r"(b2),
          "r"(b3), "r"(c0), "r"(c1), "r"(e));
      #else
    asm volatile(
        "mma.sp.sync.aligned.m16n8k64.row.col.f16.e4m3.e5m2.f16 "
        "{%0, %1}, {%2, %3, %4, %5}, {%6, %7, %8, %9}, {%10, %11}, %12, 0x0;\n"
        : "=r"(d0), "=r"(d1)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "r"(b2),
          "r"(b3), "r"(c0), "r"(c1), "r"(e));
      #endif
    #endif
  }
};

struct SM90_SPARSE_16x8x64_F16E5M2E4M3F16_TN {
  using DRegisters = uint32_t[2];
  using ARegisters = uint32_t[4];
  using BRegisters = uint32_t[4];
  using CRegisters = uint32_t[2];

  CUTE_HOST_DEVICE static void fma(uint32_t& d0, uint32_t& d1,
                                   uint32_t const& a0, uint32_t const& a1,
                                   uint32_t const& a2, uint32_t const& a3,
                                   uint32_t const& b0, uint32_t const& b1,
                                   uint32_t const& b2, uint32_t const& b3,
                                   uint32_t const& c0, uint32_t const& c1,
                                   uint32_t const& e, int const& spsel = 0) {
    #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 890
    (void)spsel;
      #if (__CUDACC_VER_MAJOR__ > 12) ||                                       \
          (__CUDACC_VER_MAJOR__ == 12 && __CUDACC_VER_MINOR__ >= 3)
    asm volatile(
        "mma.sp::ordered_metadata.sync.aligned.m16n8k64.row.col.f16.e5m2.e4m3."
        "f16 "
        "{%0, %1}, {%2, %3, %4, %5}, {%6, %7, %8, %9}, {%10, %11}, %12, 0x0;\n"
        : "=r"(d0), "=r"(d1)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "r"(b2),
          "r"(b3), "r"(c0), "r"(c1), "r"(e));
      #else
    asm volatile(
        "mma.sp.sync.aligned.m16n8k64.row.col.f16.e5m2.e4m3.f16 "
        "{%0, %1}, {%2, %3, %4, %5}, {%6, %7, %8, %9}, {%10, %11}, %12, 0x0;\n"
        : "=r"(d0), "=r"(d1)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "r"(b2),
          "r"(b3), "r"(c0), "r"(c1), "r"(e));
      #endif
    #endif
  }
};

struct SM90_SPARSE_16x8x64_F16E5M2E5M2F16_TN {
  using DRegisters = uint32_t[2];
  using ARegisters = uint32_t[4];
  using BRegisters = uint32_t[4];
  using CRegisters = uint32_t[2];

  CUTE_HOST_DEVICE static void fma(uint32_t& d0, uint32_t& d1,
                                   uint32_t const& a0, uint32_t const& a1,
                                   uint32_t const& a2, uint32_t const& a3,
                                   uint32_t const& b0, uint32_t const& b1,
                                   uint32_t const& b2, uint32_t const& b3,
                                   uint32_t const& c0, uint32_t const& c1,
                                   uint32_t const& e, int const& spsel = 0) {
    #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 890
    (void)spsel;
      #if (__CUDACC_VER_MAJOR__ > 12) ||                                       \
          (__CUDACC_VER_MAJOR__ == 12 && __CUDACC_VER_MINOR__ >= 3)
    asm volatile(
        "mma.sp::ordered_metadata.sync.aligned.m16n8k64.row.col.f16.e5m2.e5m2."
        "f16 "
        "{%0, %1}, {%2, %3, %4, %5}, {%6, %7, %8, %9}, {%10, %11}, %12, 0x0;\n"
        : "=r"(d0), "=r"(d1)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "r"(b2),
          "r"(b3), "r"(c0), "r"(c1), "r"(e));
      #else
    asm volatile(
        "mma.sp.sync.aligned.m16n8k64.row.col.f16.e5m2.e5m2.f16 "
        "{%0, %1}, {%2, %3, %4, %5}, {%6, %7, %8, %9}, {%10, %11}, %12, 0x0;\n"
        : "=r"(d0), "=r"(d1)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "r"(b2),
          "r"(b3), "r"(c0), "r"(c1), "r"(e));
      #endif
    #endif
  }
};

struct SM90_SPARSE_16x8x64_F32E4M3E4M3F32_TN {
  using DRegisters = float[4];
  using ARegisters = uint32_t[4];
  using BRegisters = uint32_t[4];
  using CRegisters = float[4];

  CUTE_HOST_DEVICE static void
  fma(float& d0, float& d1, float& d2, float& d3, uint32_t const& a0,
      uint32_t const& a1, uint32_t const& a2, uint32_t const& a3,
      uint32_t const& b0, uint32_t const& b1, uint32_t const& b2,
      uint32_t const& b3, float const& c0, float const& c1, float const& c2,
      float const& c3, uint32_t const& e, int const& spsel = 0) {
    #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 890
    (void)spsel;
      #if (__CUDACC_VER_MAJOR__ > 12) ||                                       \
          (__CUDACC_VER_MAJOR__ == 12 && __CUDACC_VER_MINOR__ >= 3)
    asm volatile("mma.sp::ordered_metadata.sync.aligned.m16n8k64.row.col.f32."
                 "e4m3.e4m3.f32 "
                 "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9, %10, %11}, "
                 "{%12, %13, %14, %15}, %16, 0x0;\n"
                 : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1),
                   "r"(b2), "r"(b3), "f"(c0), "f"(c1), "f"(c2), "f"(c3),
                   "r"(e));
      #else
    asm volatile("mma.sp.sync.aligned.m16n8k64.row.col.f32.e4m3.e4m3.f32 "
                 "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9, %10, %11}, "
                 "{%12, %13, %14, %15}, %16, 0x0;\n"
                 : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1),
                   "r"(b2), "r"(b3), "f"(c0), "f"(c1), "f"(c2), "f"(c3),
                   "r"(e));
      #endif
    #endif
  }
};

struct SM90_SPARSE_16x8x64_F32E4M3E5M2F32_TN {
  using DRegisters = float[4];
  using ARegisters = uint32_t[4];
  using BRegisters = uint32_t[4];
  using CRegisters = float[4];

  CUTE_HOST_DEVICE static void
  fma(float& d0, float& d1, float& d2, float& d3, uint32_t const& a0,
      uint32_t const& a1, uint32_t const& a2, uint32_t const& a3,
      uint32_t const& b0, uint32_t const& b1, uint32_t const& b2,
      uint32_t const& b3, float const& c0, float const& c1, float const& c2,
      float const& c3, uint32_t const& e, int const& spsel = 0) {
    #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 890
    (void)spsel;
      #if (__CUDACC_VER_MAJOR__ > 12) ||                                       \
          (__CUDACC_VER_MAJOR__ == 12 && __CUDACC_VER_MINOR__ >= 3)
    asm volatile("mma.sp::ordered_metadata.sync.aligned.m16n8k64.row.col.f32."
                 "e4m3.e5m2.f32 "
                 "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9, %10, %11}, "
                 "{%12, %13, %14, %15}, %16, 0x0;\n"
                 : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1),
                   "r"(b2), "r"(b3), "f"(c0), "f"(c1), "f"(c2), "f"(c3),
                   "r"(e));
      #else
    asm volatile("mma.sp.sync.aligned.m16n8k64.row.col.f32.e4m3.e5m2.f32 "
                 "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9, %10, %11}, "
                 "{%12, %13, %14, %15}, %16, 0x0;\n"
                 : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1),
                   "r"(b2), "r"(b3), "f"(c0), "f"(c1), "f"(c2), "f"(c3),
                   "r"(e));
      #endif
    #endif
  }
};

struct SM90_SPARSE_16x8x64_F32E5M2E4M3F32_TN {
  using DRegisters = float[4];
  using ARegisters = uint32_t[4];
  using BRegisters = uint32_t[4];
  using CRegisters = float[4];

  CUTE_HOST_DEVICE static void
  fma(float& d0, float& d1, float& d2, float& d3, uint32_t const& a0,
      uint32_t const& a1, uint32_t const& a2, uint32_t const& a3,
      uint32_t const& b0, uint32_t const& b1, uint32_t const& b2,
      uint32_t const& b3, float const& c0, float const& c1, float const& c2,
      float const& c3, uint32_t const& e, int const& spsel = 0) {
    #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 890
    (void)spsel;
      #if (__CUDACC_VER_MAJOR__ > 12) ||                                       \
          (__CUDACC_VER_MAJOR__ == 12 && __CUDACC_VER_MINOR__ >= 3)
    asm volatile("mma.sp::ordered_metadata.sync.aligned.m16n8k64.row.col.f32."
                 "e5m2.e4m3.f32 "
                 "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9, %10, %11}, "
                 "{%12, %13, %14, %15}, %16, 0x0;\n"
                 : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1),
                   "r"(b2), "r"(b3), "f"(c0), "f"(c1), "f"(c2), "f"(c3),
                   "r"(e));
      #else
    asm volatile("mma.sp.sync.aligned.m16n8k64.row.col.f32.e5m2.e4m3.f32 "
                 "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9, %10, %11}, "
                 "{%12, %13, %14, %15}, %16, 0x0;\n"
                 : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1),
                   "r"(b2), "r"(b3), "f"(c0), "f"(c1), "f"(c2), "f"(c3),
                   "r"(e));
      #endif
    #endif
  }
};

struct SM90_SPARSE_16x8x64_F32E5M2E5M2F32_TN {
  using DRegisters = float[4];
  using ARegisters = uint32_t[4];
  using BRegisters = uint32_t[4];
  using CRegisters = float[4];

  CUTE_HOST_DEVICE static void
  fma(float& d0, float& d1, float& d2, float& d3, uint32_t const& a0,
      uint32_t const& a1, uint32_t const& a2, uint32_t const& a3,
      uint32_t const& b0, uint32_t const& b1, uint32_t const& b2,
      uint32_t const& b3, float const& c0, float const& c1, float const& c2,
      float const& c3, uint32_t const& e, int const& spsel = 0) {
    #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 890
    (void)spsel;
      #if (__CUDACC_VER_MAJOR__ > 12) ||                                       \
          (__CUDACC_VER_MAJOR__ == 12 && __CUDACC_VER_MINOR__ >= 3)
    asm volatile("mma.sp::ordered_metadata.sync.aligned.m16n8k64.row.col.f32."
                 "e5m2.e5m2.f32 "
                 "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9, %10, %11}, "
                 "{%12, %13, %14, %15}, %16, 0x0;\n"
                 : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1),
                   "r"(b2), "r"(b3), "f"(c0), "f"(c1), "f"(c2), "f"(c3),
                   "r"(e));
      #else
    asm volatile("mma.sp.sync.aligned.m16n8k64.row.col.f32.e5m2.e5m2.f32 "
                 "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9, %10, %11}, "
                 "{%12, %13, %14, %15}, %16, 0x0;\n"
                 : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1),
                   "r"(b2), "r"(b3), "f"(c0), "f"(c1), "f"(c2), "f"(c3),
                   "r"(e));
      #endif
    #endif
  }
};
  #endif // __CHOREO_TARGET_NATIVE_FP8_SUPPORT__

} // namespace cute

namespace choreo {
namespace nv_cute {

namespace numerics {

template <typename T>
using DecayT = std::remove_cv_t<std::remove_reference_t<T>>;

template <typename T>
CUTE_HOST_DEVICE auto to_math(T value) {
  using U = DecayT<T>;
  if constexpr (std::is_same_v<U, f64>) {
    return static_cast<double>(value);
  } else {
    return static_cast<float>(to_f32(value));
  }
}

template <typename T, typename U>
CUTE_HOST_DEVICE T from_math(U value) {
  if constexpr (std::is_same_v<T, f64>) {
    return static_cast<f64>(value);
  } else if constexpr (std::is_same_v<T, f32>) {
    return static_cast<f32>(value);
  } else {
    return utils::from_f32<T>(static_cast<float>(value));
  }
}

  #define CHOREO_CUTE_UNARY_NUMERIC(NAME, F32, F64)                            \
    template <typename T>                                                      \
    CUTE_HOST_DEVICE auto NAME(T value) {                                      \
      using U = DecayT<T>;                                                     \
      if constexpr (std::is_same_v<U, f64>) {                                  \
        return from_math<U>(F64(to_math(value)));                              \
      } else {                                                                 \
        return from_math<U>(F32(to_math(value)));                              \
      }                                                                        \
    }

  #define CHOREO_CUTE_BINARY_NUMERIC(NAME, F32, F64)                           \
    template <typename A, typename B>                                          \
    CUTE_HOST_DEVICE auto NAME(A lhs, B rhs) {                                 \
      using R = DecayT<A>;                                                     \
      if constexpr (std::is_same_v<DecayT<A>, f64> ||                          \
                    std::is_same_v<DecayT<B>, f64>) {                          \
        return from_math<R>(F64(to_math(lhs), to_math(rhs)));                  \
      } else {                                                                 \
        return from_math<R>(F32(to_math(lhs), to_math(rhs)));                  \
      }                                                                        \
    }

CHOREO_CUTE_UNARY_NUMERIC(acos, ::acosf, ::acos);
CHOREO_CUTE_UNARY_NUMERIC(asin, ::asinf, ::asin);
CHOREO_CUTE_UNARY_NUMERIC(atan, ::atanf, ::atan);
CHOREO_CUTE_UNARY_NUMERIC(ceil, ::ceilf, ::ceil);
CHOREO_CUTE_UNARY_NUMERIC(cos, ::cosf, ::cos);
CHOREO_CUTE_UNARY_NUMERIC(cosh, ::coshf, ::cosh);
CHOREO_CUTE_UNARY_NUMERIC(exp, ::expf, ::exp);
CHOREO_CUTE_UNARY_NUMERIC(exp2f, ::exp2f, ::exp2);
CHOREO_CUTE_UNARY_NUMERIC(expm1, ::expm1f, ::expm1);
CHOREO_CUTE_UNARY_NUMERIC(fabs, ::fabsf, ::fabs);
CHOREO_CUTE_UNARY_NUMERIC(floor, ::floorf, ::floor);
CHOREO_CUTE_UNARY_NUMERIC(log, ::logf, ::log);
CHOREO_CUTE_UNARY_NUMERIC(log1p, ::log1pf, ::log1p);
CHOREO_CUTE_UNARY_NUMERIC(round, ::roundf, ::round);
CHOREO_CUTE_UNARY_NUMERIC(sin, ::sinf, ::sin);
CHOREO_CUTE_UNARY_NUMERIC(sinh, ::sinhf, ::sinh);
CHOREO_CUTE_UNARY_NUMERIC(sqrt, ::sqrtf, ::sqrt);
CHOREO_CUTE_UNARY_NUMERIC(tan, ::tanf, ::tan);
CHOREO_CUTE_UNARY_NUMERIC(tanh, ::tanhf, ::tanh);

CHOREO_CUTE_BINARY_NUMERIC(atan2, ::atan2f, ::atan2);
CHOREO_CUTE_BINARY_NUMERIC(max, ::fmaxf, ::fmax);
CHOREO_CUTE_BINARY_NUMERIC(min, ::fminf, ::fmin);
CHOREO_CUTE_BINARY_NUMERIC(pow, ::powf, ::pow);

template <typename T>
CUTE_HOST_DEVICE auto rsqrt(T value) {
  using U = DecayT<T>;
  auto x = to_math(value);
  return from_math<U>(decltype(x)(1) /
                      (std::is_same_v<U, f64> ? ::sqrt(x) : ::sqrtf(x)));
}

template <typename T>
CUTE_HOST_DEVICE auto sigmoid(T value) {
  using U = DecayT<T>;
  auto x = to_math(value);
  auto one = decltype(x)(1);
  auto y = one / (one + (std::is_same_v<U, f64> ? ::exp(-x) : ::expf(-x)));
  return from_math<U>(y);
}

template <typename T>
CUTE_HOST_DEVICE auto softplus(T value) {
  using U = DecayT<T>;
  auto x = to_math(value);
  auto y = std::is_same_v<U, f64> ? ::log1p(::exp(x)) : ::log1pf(::expf(x));
  return from_math<U>(y);
}

template <typename T>
CUTE_HOST_DEVICE auto gelu(T value) {
  using U = DecayT<T>;
  auto x = to_math(value);
  auto k0 = std::is_same_v<U, f64> ? 0.044715 : 0.044715f;
  auto k1 = std::is_same_v<U, f64> ? 0.7978845608028654 : 0.7978845834732056f;
  auto y = x * (std::is_same_v<U, f64> ? 0.5 : 0.5f) *
           ((std::is_same_v<U, f64> ? 1.0 : 1.0f) +
            (std::is_same_v<U, f64> ? ::tanh(k1 * (x + k0 * x * x * x))
                                    : ::tanhf(k1 * (x + k0 * x * x * x))));
  return from_math<U>(y);
}

template <typename T>
CUTE_HOST_DEVICE auto sign(T value) {
  using U = DecayT<T>;
  auto x = to_math(value);
  return from_math<U>((x > 0) - (x < 0));
}

template <typename T>
CUTE_HOST_DEVICE auto isfinite(T value) {
  using U = DecayT<T>;
  auto x = to_math(value);
  return from_math<U>(::isfinite(x) ? 1 : 0);
}

  #undef CHOREO_CUTE_UNARY_NUMERIC
  #undef CHOREO_CUTE_BINARY_NUMERIC

} // namespace numerics

// warp-level operation implementation
namespace warp_cooperative {

constexpr int _WARP_SIZE = 32;

template <typename FragTy, typename FTy>
__device__ __attribute__((always_inline)) inline void
fragment_elementwise(FragTy& frag, FTy&& f) {
  #pragma unroll
  for (int i = 0; i < frag.num_elements; ++i)
    frag.x[i] = f(frag.x[i]); // each lane updates its own elements
}

template <typename FragTy, typename ETy, typename FTy>
__device__ __attribute__((always_inline)) inline void
fragment_scalar_elementwise(FragTy& frag, const ETy& s, FTy&& f) {
  #pragma unroll
  for (int i = 0; i < frag.num_elements; ++i)
    frag.x[i] = f(frag.x[i], s); // each lane updates its own elements
}

template <typename FragTy, typename ETy, typename FTy>
__device__ __attribute__((always_inline)) inline void
fragment_scalarx2_elementwise(FragTy& frag, const ETy& s0, const ETy& s1,
                              FTy&& f) {
  #pragma unroll
  for (int i = 0; i < frag.num_elements; ++i)
    frag.x[i] = f(frag.x[i], s0, s1); // each lane updates its own elements
}

template <int M, int N, int LD /*leading dim*/, typename ETy, typename FTy>
__device__ __attribute__((always_inline)) inline void
inplace_matrix_uop(ETy __restrict__* m, FTy&& f) {
  int lane;
  asm("mov.u32 %0, %laneid;" : "=r"(lane));

  int MN = M * N;
  #pragma unroll
  for (int idx = lane; idx < MN; idx += _WARP_SIZE) {
    int r = idx / N;
    m[r * LD + (idx - r * N)] = f(m[r * LD + (idx - r * N)]);
  }
}

template <int M, int N, int LD /*leading dim*/, typename ETy, typename FTy>
__device__ __attribute__((always_inline)) inline void
inplace_matrix_vector_bop(ETy __restrict__* m, const ETy __restrict__* v0,
                          FTy&& f) {
  int lane;
  asm("mov.u32 %0, %laneid;" : "=r"(lane));

  int MN = M * N;
  #pragma unroll
  for (int idx = lane; idx < MN; idx += _WARP_SIZE) {
    int r = idx / N;
    int c = idx - r * N;
    m[r * LD + c] = f(m[r * LD + c], v0[r]);
  }
}

template <int M, int N, int LD /*leading dim*/, typename ETy, typename FTy>
__device__ __attribute__((always_inline)) inline void
inplace_matrix_vectorx2_bop(ETy __restrict__* m, const ETy __restrict__* v0,
                            const ETy __restrict__* v1, FTy&& f) {
  int lane;
  asm("mov.u32 %0, %laneid;" : "=r"(lane));

  int MN = M * N;
  #pragma unroll
  for (int idx = lane; idx < MN; idx += _WARP_SIZE) {
    int r = idx / N;
    int c = idx - r * N;
    m[r * LD + c] = f(m[r * LD + c], v0[r], v1[r]);
  }
}

} // end namespace warp_cooperative

} // end namespace nv_cute

// ---------------------------------------------------------------------------
// AllReduce -- cross-thread butterfly reduction for fragment reductions.
// Modeled after TileLang's AllReduce template (reduce.h).
//
// Template parameters:
//   Reducer  - binary reduction functor (MaxOp, SumOp).
//   threads  - number of threads spanning the reduce dimension.
//   scale    - stride between participating threads (default 1).
//
// Uses __shfl_xor_sync for intra-warp rounds and shared memory for
// cross-warp rounds (offset >= 32).
// ---------------------------------------------------------------------------

struct MaxOp {
  template <typename T>
  __device__ __forceinline__ T operator()(T x, T y) const {
    return x > y ? x : y;
  }
};

struct SumOp {
  template <typename T>
  __device__ __forceinline__ T operator()(T x, T y) const {
    return x + y;
  }
};

template <class Reducer, int threads, int scale = 1>
struct AllReduce {
  static_assert(threads % scale == 0, "threads must be divisible by scale");

  template <typename T>
  static __device__ __forceinline__ T run(T x, T* red_buf = nullptr) {
    if constexpr (threads == scale) {
      return x;
    } else {
      return butterfly(x, red_buf);
    }
  }

private:
  using Next = AllReduce<Reducer, threads / 2, scale>;

  template <typename T>
  static __device__ __forceinline__ T butterfly(T x, T* red_buf) {
    constexpr int offset = threads / 2;
    if constexpr (offset >= 32) {
      __syncthreads();
      red_buf[threadIdx.x] = x;
      __syncthreads();
      x = Reducer()(x, red_buf[threadIdx.x ^ offset]);
    } else {
      x = Reducer()(x, __shfl_xor_sync(0xffffffff, x, offset));
    }
    if constexpr (offset == scale) {
      return x;
    } else {
      return Next::run(x, red_buf);
    }
  }
};

} // end namespace choreo

#endif // __CHOREO_TARGET_CUTE__

#endif //__CHROEO_CUTE_HEADER_LIBRARY_HPP__

EOF

cat <<'EOF' > /tmp/1780994114377551139_138024178890560/__choreo_cute_debug_rtti_GDB_types.cu

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <iterator>
#include <string>
#include <vector>

#include "cutlass/cutlass.h"

#ifdef __CUDACC__
#pragma nv_diag_suppress 20054
#endif
// include the choreo header;
#define __CHOREO_ENABLE_CUDA_RUNTIME_ENV_CHECK__
#define __CHOREO_REQUIRED_GPU_DEVICE_SM__ 86
#include "choreo.h"


static __device__ __attribute__((noinline)) void __choreo_cuda_debug_point__() {
  asm volatile("" ::: "memory");
}
#include <cooperative_groups.h>
using namespace choreo;





__global__ void __choreo_device_debug_builtin_non_scalar_types(int * lhs, int x) {
  [[maybe_unused]] auto __choreo_device_debug_builtin_non_scalar_types__ring__ = nullptr;
  #line 13 "tests/gpu/end2end/debug_rtti_GDB_types.co"
#line 13 "tests/gpu/end2end/debug_rtti_GDB_types.co"
  if (__CHOREO_BLOCK_SINGLE__) __choreo_cuda_debug_point__();
#line 13 "tests/gpu/end2end/debug_rtti_GDB_types.co"
  { // parallel-by: tests/gpu/end2end/debug_rtti_GDB_types.co:13.12
#line 14 "tests/gpu/end2end/debug_rtti_GDB_types.co"
#line 14 "tests/gpu/end2end/debug_rtti_GDB_types.co"
  // with-in: tests/gpu/end2end/debug_rtti_GDB_types.co:14.5
#line 14 "tests/gpu/end2end/debug_rtti_GDB_types.co"
  {
#line 14 "tests/gpu/end2end/debug_rtti_GDB_types.co"
    int __iv_didx__elem__0 = 0;
#line 14 "tests/gpu/end2end/debug_rtti_GDB_types.co"
    int __iv_didx__elem__1 = 0;
#line 14 "tests/gpu/end2end/debug_rtti_GDB_types.co"
    int __iv_didx__elem__2 = 0;
#line 14 "tests/gpu/end2end/debug_rtti_GDB_types.co"
    choreo::rtti::bounded_ituple<3> __iv_didx = {{__iv_didx__elem__0, __iv_didx__elem__1, __iv_didx__elem__2}, {1, x, 3}};
#line 14 "tests/gpu/end2end/debug_rtti_GDB_types.co"
    auto didx = __iv_didx;
#line 15 "tests/gpu/end2end/debug_rtti_GDB_types.co"
#line 15 "tests/gpu/end2end/debug_rtti_GDB_types.co"
    printf("%lld{%lld, %lld, %lld}\n", static_cast<long long>(222), static_cast<long long>(__iv_didx[0]), static_cast<long long>(__iv_didx[1]), static_cast<long long>(__iv_didx[2]));
#line 15 "tests/gpu/end2end/debug_rtti_GDB_types.co"
  }
#line 13 "tests/gpu/end2end/debug_rtti_GDB_types.co"
#line 13 "tests/gpu/end2end/debug_rtti_GDB_types.co"
  } // end parallel-by
#line 13 "tests/gpu/end2end/debug_rtti_GDB_types.co"
}

#line 13 "tests/gpu/end2end/debug_rtti_GDB_types.co"
void debug_builtin_non_scalar_types(const choreo::spanned_view<choreo::s32, 2> & lhs, int x) {
#line 13 "tests/gpu/end2end/debug_rtti_GDB_types.co"
  __choreo_check_cuda_environment__();
#line 13 "tests/gpu/end2end/debug_rtti_GDB_types.co"
  choreo::runtime_check(lhs.shape()[0] == 32, "shape inconsistent on the 1st parameter ('lhs', dim: 0): expect: 32, but got " + std::to_string(lhs.shape()[0]) + ".");
#line 13 "tests/gpu/end2end/debug_rtti_GDB_types.co"
  choreo::runtime_check(lhs.shape()[1] == 64, "shape inconsistent on the 1st parameter ('lhs', dim: 1): expect: 64, but got " + std::to_string(lhs.shape()[1]) + ".");

#line 13 "tests/gpu/end2end/debug_rtti_GDB_types.co"
  choreo::runtime_check((static_cast<long long>(x) != 0LL), "zero is detected for the 2nd dim of the mdspan inside the with-in statement, tests/gpu/end2end/debug_rtti_GDB_types.co:9.15");
#line 13 "tests/gpu/end2end/debug_rtti_GDB_types.co"
  choreo::runtime_check((static_cast<long long>(x) != 0LL), "zero is detected for the 2nd dim of the mdspan inside the with-in statement, tests/gpu/end2end/debug_rtti_GDB_types.co:14.18");
#line 13 "tests/gpu/end2end/debug_rtti_GDB_types.co"
  choreo::rtti::spanned<int, 2> __dbg_lhs = {{lhs.shape()[0], lhs.shape()[1]}, {64, 1}, lhs.data()};
#line 13 "tests/gpu/end2end/debug_rtti_GDB_types.co"
  int * lhs__device = nullptr;
#line 13 "tests/gpu/end2end/debug_rtti_GDB_types.co"
  choreo::abend_true(cudaMalloc(&lhs__device, 8192ULL));
#line 13 "tests/gpu/end2end/debug_rtti_GDB_types.co"
  choreo::abend_true(cudaMemcpy(lhs__device, lhs.data(), 8192ULL, cudaMemcpyHostToDevice));
#line 13 "tests/gpu/end2end/debug_rtti_GDB_types.co"
  choreo::rtti::mdspan<3> a = {7, 8, 9};
#line 7 "tests/gpu/end2end/debug_rtti_GDB_types.co"
#line 7 "tests/gpu/end2end/debug_rtti_GDB_types.co"
  choreo::rtti::ituple<3> b = {1, x, 3};
#line 9 "tests/gpu/end2end/debug_rtti_GDB_types.co"
#line 9 "tests/gpu/end2end/debug_rtti_GDB_types.co"
  // with-in: tests/gpu/end2end/debug_rtti_GDB_types.co:9.3
#line 9 "tests/gpu/end2end/debug_rtti_GDB_types.co"
  {
#line 9 "tests/gpu/end2end/debug_rtti_GDB_types.co"
    int __iv_idx__elem__0 = 0;
#line 9 "tests/gpu/end2end/debug_rtti_GDB_types.co"
    int __iv_idx__elem__1 = 0;
#line 9 "tests/gpu/end2end/debug_rtti_GDB_types.co"
    int __iv_idx__elem__2 = 0;
#line 9 "tests/gpu/end2end/debug_rtti_GDB_types.co"
    choreo::rtti::bounded_ituple<3> __iv_idx = {{__iv_idx__elem__0, __iv_idx__elem__1, __iv_idx__elem__2}, {1, x, 3}};
#line 9 "tests/gpu/end2end/debug_rtti_GDB_types.co"
    auto idx = __iv_idx;
#line 10 "tests/gpu/end2end/debug_rtti_GDB_types.co"
#line 10 "tests/gpu/end2end/debug_rtti_GDB_types.co"
    printf("%lld{%lld, %lld, %lld}\n", static_cast<long long>(111), static_cast<long long>(idx[0]), static_cast<long long>(idx[1]), static_cast<long long>(idx[2]));
#line 10 "tests/gpu/end2end/debug_rtti_GDB_types.co"
  }
#line 13 "tests/gpu/end2end/debug_rtti_GDB_types.co"
#line 13 "tests/gpu/end2end/debug_rtti_GDB_types.co"
  dim3 __debug_builtin_non_scalar_types_gdims0(1, 1, 1);
#line 13 "tests/gpu/end2end/debug_rtti_GDB_types.co"
  dim3 __debug_builtin_non_scalar_types_bdims0(1, 1, 1);
#line 13 "tests/gpu/end2end/debug_rtti_GDB_types.co"
  __choreo_device_debug_builtin_non_scalar_types<<<__debug_builtin_non_scalar_types_gdims0, __debug_builtin_non_scalar_types_bdims0>>>(lhs__device, x);
#line 13 "tests/gpu/end2end/debug_rtti_GDB_types.co"
  choreo::abend_true(cudaDeviceSynchronize());
#line 13 "tests/gpu/end2end/debug_rtti_GDB_types.co"
  choreo::abend_true(cudaFree(lhs__device));
#line 13 "tests/gpu/end2end/debug_rtti_GDB_types.co"
}




int main() {
  auto lhs = choreo::make_spandata<choreo::s32>(32, 64);
  lhs.fill(0);
  debug_builtin_non_scalar_types(lhs.view(), 1);
  std::cout << "Run Done\n";
  return 0;
}




























EOF

nv_arch=sm_86

show_usage() {
  echo "  Usage: $0 <actions>"
  echo ""
  echo "  Options:"
  echo "   --execute,           Compile and execute"
  echo "   --compile-link,      Compile and link"
  echo "   --compile-module,    Compile and generate the module"
  echo "   --gen-fatbin,        Compile and generate the fatbin"
  echo "   --lib,               Compile and generate the lib"
  echo ""
  echo "  Environment Variables:"
  echo "   CUDA_HOME:           (Must) Cuda compiler installation path"
  echo "   CUTE_HOME:           (Must) Cute header library path"
  echo "   EXTRA_TARGET_CFLAGS: Extra target compilation flags"
  exit 1
}

# compile, execute
export CFLAGS="-arch ${nv_arch} -std=c++17 -DCUTLASS_ENABLE_TENSOR_CORE_MMA=1 -D__CHOREO_TARGET_CUTE__ -Xcompiler -static-libstdc++ -lcuda -O0 -D__USE_CUDA_TYPE__ -g -G --use_fast_math -D__CHOREO_DMA_DIAGNOSIS__ --use_fast_math --expt-relaxed-constexpr ${EXTRA_TARGET_CFLAGS} -I/home/fem/dev/choreo/tests/gpu/end2end -I${CUTE_HOME}/include -L${CUDA_HOME}/lib64 -lcuda"
export LD_LIBRARY_PATH=${CUDA_LIB}:${LD_LIBRARY_PATH}

if [ "$1" == "--execute" ] || [ "$#" -eq 0 ]; then
  ${NVCC} ${CFLAGS} /tmp/1780994114377551139_138024178890560/__choreo_cute_debug_rtti_GDB_types.cu -o /tmp/1780994114377551139_138024178890560/__choreo_cute_debug_rtti_GDB_types.exe
  /tmp/1780994114377551139_138024178890560/__choreo_cute_debug_rtti_GDB_types.exe
elif [ "$1" == "--compile-module" ]; then
  ${NVCC} -c ${CFLAGS} /tmp/1780994114377551139_138024178890560/__choreo_cute_debug_rtti_GDB_types.cu -o /tmp/1780994114377551139_138024178890560/__choreo_cute_debug_rtti_GDB_types.exe
elif [ "$1" == "--compile-link" ]; then
  ${NVCC} ${CFLAGS} /tmp/1780994114377551139_138024178890560/__choreo_cute_debug_rtti_GDB_types.cu -o /tmp/1780994114377551139_138024178890560/__choreo_cute_debug_rtti_GDB_types.exe
elif [ "$1" == "--lib" ]; then
  ${NVCC} --lib -Xcompiler -fPIC ${CFLAGS} /tmp/1780994114377551139_138024178890560/__choreo_cute_debug_rtti_GDB_types.cu -o /tmp/1780994114377551139_138024178890560/__choreo_cute_debug_rtti_GDB_types.exe

else show_usage
fi