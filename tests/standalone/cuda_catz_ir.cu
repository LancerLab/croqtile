
#include <gtest/gtest.h>

#include <iostream>
#include <algorithm>
#include <fstream>
#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#include "utils/catz/macro.h"
#include "utils/catz/index.h"
#include "utils/catz/trait.h"

// TODO: move to testing utils
namespace { // some testing utils for CUDA testing only

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

}

class CatzIRTest : public ::testing::Test {
 protected:
  std::string temp_filename;
  int tempfile_desc;

  virtual void SetUp() {
    char temp_template[] = "/tmp/optiontest-XXXXXX";
    tempfile_desc = mkstemp(temp_template);
    if (tempfile_desc == -1) {
      perror("Error creating temporary file");
      exit(EXIT_FAILURE);
    }
    temp_filename = temp_template;  // Update filename
  }

  virtual void TearDown() {
    close(tempfile_desc);           // Close file descriptor
    remove(temp_filename.c_str());  // Delete file
  }

  void createFileWithContent(const std::string& filename,
                             const std::string& content) {
    std::ofstream out(filename);
    ASSERT_TRUE(out.is_open());
    out << content;
    out.close();
  }
};

TEST_F(CatzIRTest, TraitTest) {
  const int compileTimeValue = 42;
  auto indexConst = make_index<compileTimeValue>();
  SCHECK(is_index_like_v<decltype(indexConst)> == true);

  int runTimeValue = 42;
  auto indexDyn = make_index(runTimeValue); // 自动使用基础版本
  SCHECK(is_index_like_v<int> == false);
  SCHECK(is_index_like_v<decltype(indexDyn)> == true);
}
