// clang-format off
#define CHECK(cond) ASSERT_TRUE(cond)
#define CHECK_FALSE(cond) ASSERT_FALSE(cond)
#define SCHECK(cond)                                                           \
  static_assert(cond, "compile-time assertion check failed\n")
#define SCHECK_FALSE(cond) static_assert(!(cond), "compile-time check failed\n")

#define CONCATENATE(x, y) x##y

#define CUDA_KERNEL_NAME(name) CONCATENATE(name, _cuda)

// Macro to check CUDA errors
#define CUDA_CHECK_ERROR()                                                                                   \
  \                                                                                                       
  do {                                                                                                       \
    \                                                                                                       
    cudaError_t err = cudaGetLastError();                                                                    \
    \                                                                                                       
    CHECK(err == cudaSuccess);                                                                               \
    \                                                                                                       
    cudaDeviceSynchronize();                                                                                 \
    \                                                                                                       
    \
  }                                                                                                          \
  while (0)

#define TEST_CUDA_F(group_name, case_name)                                                                   \
  \                                                                                                       
  __global__ void CUDA_KERNEL_NAME(case_name)();                                                             \
  \                                                                                                       
  TEST_F(group_name, case_name) {                                                                            \
    \                                                                                                       
    CUDA_KERNEL_NAME(case_name)<<<1, 1>>>();                                                                 \
    \                                                                                                       
    CUDA_CHECK_ERROR();                                                                                      \
    \                                                                                                       
    \
  }                                                                                                          \
  \                                                                                                       
  __global__ void                                                                                            \
  CUDA_KERNEL_NAME(case_name)()

#define TEST_CUDA_F_THREADS(group_name, case_name, num_of_threads)                                           \
  \                                                                                                       
  __global__ void CUDA_KERNEL_NAME(case_name)();                                                             \
  \                                                                                                       
  TEST_CASE(group_name, case_name) {                                                                         \
    \                                                                                                       
    CUDA_KERNEL_NAME(case_name)<<<1, num_of_threads>>>();                                                    \
    \                                                                                                       
    CUDA_CHECK_ERROR();                                                                                      \
    \                                                                                                       
    \
  }                                                                                                          \
  \                                                                                                       
  __global__ void                                                                                            \
  CUDA_KERNEL_NAME(case_name)()
// clang-format on
