#ifndef __CHROEO_CUTE_HEADER_LIBRARY_HPP__
#define __CHROEO_CUTE_HEADER_LIBRARY_HPP__

// Note: No manual inclusion. (included by "choreo.h")

#ifdef __CHOREO_TARGET_CUTE__

namespace choreo {
namespace nv_cute {

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
} // end namespace choreo

#endif // __CHOREO_TARGET_CUTE__

#endif //__CHROEO_CUTE_HEADER_LIBRARY_HPP__
