#ifndef CATZILLA_CATZ_CUDA_UTILS_H_
#define CATZILLA_CATZ_CUDA_UTILS_H_

#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>

namespace catz {

__device__ __forceinline__ uint get_smem_ptr(const void *ptr) {
  return static_cast<unsigned>(__cvta_generic_to_shared(ptr));
}

} // namespace catz

#endif // CATZILLA_CATZ_CUDA_UTILS_H_
