#ifndef __CHOREO_CC_RUNTIME_H__
#define __CHOREO_CC_RUNTIME_H__

#include <atomic>
#include <cstring>

namespace choreo {
namespace cc {

// -- Event support via atomic flags --
struct event_t {
  std::atomic<bool> flag{false};
  void trigger() { flag.store(true, std::memory_order_release); }
  void wait() {
    while (!flag.load(std::memory_order_acquire)) {}
    flag.store(false, std::memory_order_release);
  }
  void reset() { flag.store(false, std::memory_order_release); }
};

// -- Reference MMA (matrix multiply-accumulate) --
// D[m][n] += A[m][k] * B[k][n]  (row-major A, col-major B => row_col)
// D[m][n] += A[m][k] * B[n][k]  (row-major A, row-major B => row_row)
template <typename TD, typename TA, typename TB, typename TC>
inline void mma_exec_row_col(TD* D, const TA* A, const TB* B, const TC* C,
                             int M, int N, int K) {
  for (int m = 0; m < M; ++m)
    for (int n = 0; n < N; ++n) {
      TD acc = static_cast<TD>(C[m * N + n]);
      for (int k = 0; k < K; ++k)
        acc += static_cast<TD>(A[m * K + k]) * static_cast<TD>(B[k * N + n]);
      D[m * N + n] = acc;
    }
}

template <typename TD, typename TA, typename TB, typename TC>
inline void mma_exec_row_row(TD* D, const TA* A, const TB* B, const TC* C,
                             int M, int N, int K) {
  for (int m = 0; m < M; ++m)
    for (int n = 0; n < N; ++n) {
      TD acc = static_cast<TD>(C[m * N + n]);
      for (int k = 0; k < K; ++k)
        acc += static_cast<TD>(A[m * K + k]) * static_cast<TD>(B[n * K + k]);
      D[m * N + n] = acc;
    }
}

template <typename TD, typename TA, typename TB, typename TC>
inline void mma_exec_col_row(TD* D, const TA* A, const TB* B, const TC* C,
                             int M, int N, int K) {
  for (int m = 0; m < M; ++m)
    for (int n = 0; n < N; ++n) {
      TD acc = static_cast<TD>(C[m * N + n]);
      for (int k = 0; k < K; ++k)
        acc += static_cast<TD>(A[k * M + m]) * static_cast<TD>(B[n * K + k]);
      D[m * N + n] = acc;
    }
}

template <typename TD, typename TA, typename TB, typename TC>
inline void mma_exec_col_col(TD* D, const TA* A, const TB* B, const TC* C,
                             int M, int N, int K) {
  for (int m = 0; m < M; ++m)
    for (int n = 0; n < N; ++n) {
      TD acc = static_cast<TD>(C[m * N + n]);
      for (int k = 0; k < K; ++k)
        acc += static_cast<TD>(A[k * M + m]) * static_cast<TD>(B[k * N + n]);
      D[m * N + n] = acc;
    }
}

template <typename T>
inline void mma_fill(T* buf, T val, int count) {
  for (int i = 0; i < count; ++i) buf[i] = val;
}

template <typename T>
inline void mma_store(T* dst, const T* src, int count) {
  std::memcpy(dst, src, count * sizeof(T));
}

template <typename T>
inline void mma_store_transpose(T* dst, const T* src, int M, int N) {
  for (int m = 0; m < M; ++m)
    for (int n = 0; n < N; ++n) dst[n * M + m] = src[m * N + n];
}

template <typename TD, typename TS>
inline void mma_scale(TD* acc, const TS* scale, int count) {
  for (int i = 0; i < count; ++i)
    acc[i] =
        static_cast<TD>(static_cast<TD>(acc[i]) * static_cast<TD>(scale[i]));
}

// -- Library call: reference GEMM --
// C = alpha * A * B + beta * C
template <typename T>
inline void lib_gemm(T* C, const T* A, const T* B, int M, int N, int K,
                     T alpha = T(1), T beta = T(0)) {
  for (int m = 0; m < M; ++m)
    for (int n = 0; n < N; ++n) {
      T acc = T(0);
      for (int k = 0; k < K; ++k) acc += A[m * K + k] * B[k * N + n];
      C[m * N + n] = alpha * acc + beta * C[m * N + n];
    }
}

} // namespace cc
} // namespace choreo

#endif // __CHOREO_CC_RUNTIME_H__
