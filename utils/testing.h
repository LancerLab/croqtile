
#define CHECK(cond) ASSERT_TRUE(cond) 
#define CHECK_FALSE(cond) ASSERT_FALSE(cond) 
#define SCHECK(cond) static_assert(cond, "compile-time assertion check failed\n")
#define SCHECK_FALSE(cond) static_assert(!(cond), "compile-time check failed\n")

#define CONCATENATE(x, y) x##y                                                                                                                                                         
                                                                                                                                                                                       
#define CUDA_KERNEL_NAME(name) CONCATENATE(name, _cuda)                                                                                                                                
                                                                                                                                                                                       
// 检查 CUDA 错误的宏                                                                                                                                                                  
#define CUDA_CHECK_ERROR()                                                     \                                                                                                       
  do {                                                                         \                                                                                                       
    cudaError_t err = cudaGetLastError();                                      \                                                                                                       
    CHECK(err == cudaSuccess);                                                 \                                                                                                       
    cudaDeviceSynchronize();                                                   \                                                                                                       
  } while (0)                                                                                                                                                                          
                                                                                                                                                                                       
#define TEST_CUDA_CASE(kernel_name, case_name, tags)                           \                                                                                                       
  __global__ void CUDA_KERNEL_NAME(kernel_name)();                             \                                                                                                       
  TEST_CASE(case_name, tags) {                                                 \                                                                                                       
    CUDA_KERNEL_NAME(kernel_name)<<<1, 1>>>();                                 \                                                                                                       
    CUDA_CHECK_ERROR();                                                        \                                                                                                       
  }                                                                            \                                                                                                       
  __global__ void CUDA_KERNEL_NAME(kernel_name)()                                                                                                                                      
                                                                                                                                                                                       
#define TEST_CUDA_CASE_WITH_THREADS(kernel_name, num_of_threads, case_name,    \                                                                                                       
                                    tags)                                      \                                                                                                       
  __global__ void CUDA_KERNEL_NAME(kernel_name)();                             \                                                                                                       
  TEST_CASE(case_name, tags) {                                                 \                                                                                                       
    CUDA_KERNEL_NAME(kernel_name)<<<1, num_of_threads>>>();                    \                                                                                                       
    CUDA_CHECK_ERROR();                                                        \                                                                                                       
  }                                                                            \                                                                                                       
  __global__ void CUDA_KERNEL_NAME(kernel_name)()
