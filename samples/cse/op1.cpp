#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <string.h>  // 添加memcpy支持
// #include <future>
// #include <vector>
// #include <chrono>
// #include <iostream>
// #include <random>
// #include <vector>
#include <cuda_runtime.h>
// #include <functional>

// 测试配置常量 - 确保所有测试使用相同参数
#define TEST_LEN 1000        // 使用较大数据量以获得更稳定的时间测量
#define TEST_EMB_VEC_SIZE 64
#define TEST_BLOCK_SIZE 256
#define RANDOM_SEED 12345    // 固定随机种子确保数据一致性

// Forward declarations of async functions
void merge_emb_vec_async(float* d_output, const float* d_input, const uint64_t* d_index,
                        const size_t len, const size_t emb_vec_size, const size_t block_size,
                        cudaStream_t stream);

void fill_default_emb_vec_async(float* d_output, const float default_value, const uint64_t* d_index,
                                const size_t len, const size_t emb_vec_size, const size_t block_size,
                                cudaStream_t stream);

void decompress_emb_vec_async(const float* d_input, const uint64_t* d_index, float* d_output,
                              const size_t len, const size_t emb_vec_size, const size_t block_size,
                              cudaStream_t stream);

// Wrapper class for CUDA functions
template<typename... Args>
class CudaFunctionWrapper {
public:
    template<typename Ret>
    using FunctionType = Ret(*)(Args...);

    static FunctionType<void> getMergeFunction() {
        return merge_emb_vec_async;
    }

    static FunctionType<void> getFillDefaultFunction() {
        return fill_default_emb_vec_async;
    }

    static FunctionType<void> getDecompressFunction() {
        return decompress_emb_vec_async;
    }
};

// Explicit template instantiation
// using CudaFunctions = CudaFunctionWrapper<float*, const float*, const uint64_t*, size_t, size_t, size_t, cudaStream_t>;
using MergeFunc = void(*)(float*, const float*, const uint64_t*, size_t, size_t, size_t, cudaStream_t);
using FillDefaultFunc = void(*)(float*, float, const uint64_t*, size_t, size_t, size_t, cudaStream_t);
using DecompressFunc = void(*)(const float*, const uint64_t*, float*, size_t, size_t, size_t, cudaStream_t);


void fill_default_emb_vec_cpu_optimized(float* output_emb_vec, const float default_emb_vec,
    const uint64_t* missing_index, const size_t len,
    const size_t emb_vec_size) {
    // 使用双层循环避免除法和模运算
    for (size_t i = 0; i < len; i++) {
        size_t dst_emb_vec = missing_index[i];
        float* dst_ptr = output_emb_vec + dst_emb_vec * emb_vec_size;

        // 向量化填充 - 一次处理多个float值
        size_t j = 0;
        // 4个float为一组进行处理，提升缓存效率
        for (; j + 3 < emb_vec_size; j += 4) {
            dst_ptr[j] = default_emb_vec;
            dst_ptr[j + 1] = default_emb_vec;
            dst_ptr[j + 2] = default_emb_vec;
            dst_ptr[j + 3] = default_emb_vec;
        }
        // 处理剩余的元素
        for (; j < emb_vec_size; j++) {
            dst_ptr[j] = default_emb_vec;
        }
    }
}

void merge_emb_vec_cpu_optimized(float* output_emb_vec, const float* missing_emb_vec,
                  const uint64_t* missing_index, const size_t len,
                  const size_t emb_vec_size) {
    // 使用双层循环避免除法和模运算
    for (size_t i = 0; i < len; i++) {
        size_t dst_emb_vec = missing_index[i];
        const float* src_ptr = missing_emb_vec + i * emb_vec_size;
        float* dst_ptr = output_emb_vec + dst_emb_vec * emb_vec_size;

        // 向量化拷贝，利用内存局部性
        size_t j = 0;
        // 4个float为一组进行处理
        for (; j + 3 < emb_vec_size; j += 4) {
            dst_ptr[j] = src_ptr[j];
            dst_ptr[j + 1] = src_ptr[j + 1];
            dst_ptr[j + 2] = src_ptr[j + 2];
            dst_ptr[j + 3] = src_ptr[j + 3];
        }
        // 处理剩余的元素
        for (; j < emb_vec_size; j++) {
            dst_ptr[j] = src_ptr[j];
        }
    }
}

void decompress_emb_vec_cpu_optimized(const float* src_emb_vec, const uint64_t* src_index,
                       float* dst_emb_vec, const size_t len,
                       const size_t emb_vec_size) {
    // 对于小的emb_vec_size，简单的循环更高效
    if (emb_vec_size <= 8) {
        for (size_t i = 0; i < len; i++) {
            size_t src_idx = src_index[i];
            const float* src_ptr = src_emb_vec + src_idx * emb_vec_size;
            float* dst_ptr = dst_emb_vec + i * emb_vec_size;
            // 简单循环，避免展开开销
            for (size_t j = 0; j < emb_vec_size; j++) {
                dst_ptr[j] = src_ptr[j];
            }
        }
    } else {
        // 对于大的emb_vec_size，使用向量化拷贝
        for (size_t i = 0; i < len; i++) {
            size_t src_idx = src_index[i];
            const float* src_ptr = src_emb_vec + src_idx * emb_vec_size;
            float* dst_ptr = dst_emb_vec + i * emb_vec_size;

            // 向量化拷贝
            size_t j = 0;
            // 8个float为一组进行处理，提升缓存利用率
            for (; j + 7 < emb_vec_size; j += 8) {
                dst_ptr[j] = src_ptr[j];
                dst_ptr[j + 1] = src_ptr[j + 1];
                dst_ptr[j + 2] = src_ptr[j + 2];
                dst_ptr[j + 3] = src_ptr[j + 3];
                dst_ptr[j + 4] = src_ptr[j + 4];
                dst_ptr[j + 5] = src_ptr[j + 5];
                dst_ptr[j + 6] = src_ptr[j + 6];
                dst_ptr[j + 7] = src_ptr[j + 7];
            }
            // 处理剩余的元素
            for (; j < emb_vec_size; j++) {
                dst_ptr[j] = src_ptr[j];
            }
        }
    }
}

void fill_default_emb_vec_cpu(float* output_emb_vec, const float default_emb_vec,
    const uint64_t* missing_index, const size_t len,
    const size_t emb_vec_size) {
    for (size_t idx = 0; idx < (len * emb_vec_size); idx++) {
        size_t src_emb_vec = idx / emb_vec_size;
        size_t dst_emb_vec = missing_index[src_emb_vec];
        size_t dst_float = idx % emb_vec_size;
        output_emb_vec[dst_emb_vec * emb_vec_size + dst_float] = default_emb_vec;
    }
}

void merge_emb_vec_cpu(float* output_emb_vec, const float* missing_emb_vec,
                  const uint64_t* missing_index, const size_t len,
                  const size_t emb_vec_size) {
  for (size_t idx = 0; idx < (len * emb_vec_size); idx++) {
    size_t src_emb_vec = idx / emb_vec_size;
    size_t dst_emb_vec = missing_index[src_emb_vec];
    size_t dst_float = idx % emb_vec_size;
    output_emb_vec[dst_emb_vec * emb_vec_size + dst_float] =
        missing_emb_vec[src_emb_vec * emb_vec_size + dst_float];
  }
}

void decompress_emb_vec_cpu(const float* src_emb_vec, const uint64_t* src_index,
                       float* dst_emb_vec, const size_t len,
                       const size_t emb_vec_size) {
  for (size_t i = 0; i < len; i++) {
    for (size_t j = 0; j < emb_vec_size; j++) {
      size_t dst_idx = i;
      size_t dst_float = j;
      size_t src_idx = src_index[dst_idx];
      dst_emb_vec[dst_idx * emb_vec_size + dst_float] =
          src_emb_vec[src_idx * emb_vec_size + dst_float];
    }
  }
}

// 更高级的优化版本
void merge_emb_vec_cpu_advanced(float* output_emb_vec, const float* missing_emb_vec,
                  const uint64_t* missing_index, const size_t len,
                  const size_t emb_vec_size) {
    // 当向量大小较大时，使用memcpy会更高效
    const size_t vec_size_bytes = emb_vec_size * sizeof(float);

    if (emb_vec_size >= 16) {
        // 对于大向量，使用memcpy
        for (size_t i = 0; i < len; i++) {
            size_t dst_emb_vec = missing_index[i];
            const float* src_ptr = missing_emb_vec + i * emb_vec_size;
            float* dst_ptr = output_emb_vec + dst_emb_vec * emb_vec_size;
            memcpy(dst_ptr, src_ptr, vec_size_bytes);
        }
    } else {
        // 对于小向量，使用优化的循环展开
        merge_emb_vec_cpu_optimized(output_emb_vec, missing_emb_vec, missing_index, len, emb_vec_size);
    }
}

void fill_default_emb_vec_cpu_advanced(float* output_emb_vec, const float default_emb_vec,
    const uint64_t* missing_index, const size_t len,
    const size_t emb_vec_size) {

    if (emb_vec_size >= 8) {
        // 对于较大的向量，先创建一个模板向量，然后使用memcpy
        float* template_vec = (float*)malloc(emb_vec_size * sizeof(float));
        for (size_t j = 0; j < emb_vec_size; j++) {
            template_vec[j] = default_emb_vec;
        }

        for (size_t i = 0; i < len; i++) {
            size_t dst_emb_vec = missing_index[i];
            float* dst_ptr = output_emb_vec + dst_emb_vec * emb_vec_size;
            memcpy(dst_ptr, template_vec, emb_vec_size * sizeof(float));
        }

        free(template_vec);
    } else {
        // 对于小向量，使用优化的循环展开
        fill_default_emb_vec_cpu_optimized(output_emb_vec, default_emb_vec, missing_index, len, emb_vec_size);
    }
}

void decompress_emb_vec_cpu_advanced(const float* src_emb_vec, const uint64_t* src_index,
                       float* dst_emb_vec, const size_t len,
                       const size_t emb_vec_size) {
    const size_t vec_size_bytes = emb_vec_size * sizeof(float);

    if (emb_vec_size >= 16) {
        // 对于大向量，使用memcpy
        for (size_t i = 0; i < len; i++) {
            size_t src_idx = src_index[i];
            const float* src_ptr = src_emb_vec + src_idx * emb_vec_size;
            float* dst_ptr = dst_emb_vec + i * emb_vec_size;
            memcpy(dst_ptr, src_ptr, vec_size_bytes);
        }
    } else {
        // 对于小向量，使用优化的循环展开
        decompress_emb_vec_cpu_optimized(src_emb_vec, src_index, dst_emb_vec, len, emb_vec_size);
    }
}

#define CHECK_CUDA(call) \
  do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
      /* std::cerr << "CUDA error in file '" << __FILE__ << "' line " << __LINE__ << ": " \
                << cudaGetErrorString(err) << std::endl; */ \
      exit(EXIT_FAILURE); \
    } \
  } while(0)

__global__ void merge_emb_vec_kernel(float* d_output_emb_vec, const float* d_missing_emb_vec,
                              const uint64_t* d_missing_index, const size_t len,
                              const size_t emb_vec_size) {
  const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < (len * emb_vec_size)) {
    size_t src_emb_vec = idx / emb_vec_size;
    size_t dst_emb_vec = d_missing_index[src_emb_vec];
    size_t dst_float = idx % emb_vec_size;
    d_output_emb_vec[dst_emb_vec * emb_vec_size + dst_float] =
        d_missing_emb_vec[src_emb_vec * emb_vec_size + dst_float];
  }
}

__global__ void fill_default_emb_vec_kernel(float* d_output_emb_vec, const float default_emb_vec,
                                     const uint64_t* d_missing_index, const size_t len,
                                     const size_t emb_vec_size) {
  const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < (len * emb_vec_size)) {
    size_t src_emb_vec = idx / emb_vec_size;
    size_t dst_emb_vec = d_missing_index[src_emb_vec];
    size_t dst_float = idx % emb_vec_size;
    d_output_emb_vec[dst_emb_vec * emb_vec_size + dst_float] = default_emb_vec;
  }
}

__global__ void decompress_emb_vec_kernel(const float* d_src_emb_vec, const uint64_t* d_src_index,
                                   float* d_dst_emb_vec, const size_t len,
                                   const size_t emb_vec_size) {
  const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < (len * emb_vec_size)) {
    size_t dst_emb_vec = idx / emb_vec_size;
    size_t dst_float = idx % emb_vec_size;
    size_t src_emb_vec = d_src_index[dst_emb_vec];
    d_dst_emb_vec[dst_emb_vec * emb_vec_size + dst_float] =
        d_src_emb_vec[src_emb_vec * emb_vec_size + dst_float];
  }
}

void merge_emb_vec_async(float* d_output, const float* d_input, const uint64_t* d_index,
                        const size_t len, const size_t emb_vec_size, const size_t block_size,
                        cudaStream_t stream) {
  if (len == 0) {
    return;
  }
  size_t missing_len_in_float = len * emb_vec_size;

  const size_t grid_size = (missing_len_in_float - 1) / block_size + 1;
  // const size_t grid_size = (len * emb_vec_size + block_size - 1) / block_size;
  merge_emb_vec_kernel<<<grid_size, block_size, 0, stream>>>(d_output, d_input, d_index, len, emb_vec_size);
}

// void merge_emb_vec_async(float* d_vals_merge_dst_ptr, const float* d_vals_retrieved_ptr,
//   const uint64_t* d_missing_index_ptr, const size_t missing_len,
//   const size_t emb_vec_size, const size_t BLOCK_SIZE, cudaStream_t stream) {
//   if (missing_len == 0) {
//     return;
//   }
//   size_t missing_len_in_float = missing_len * emb_vec_size;
//   merge_emb_vec<<<((missing_len_in_float - 1) / BLOCK_SIZE) + 1, BLOCK_SIZE, 0, stream>>>(
//   d_vals_merge_dst_ptr, d_vals_retrieved_ptr, d_missing_index_ptr, missing_len, emb_vec_size);
// }

void fill_default_emb_vec_async(float* d_output, const float default_value, const uint64_t* d_index,
                                const size_t len, const size_t emb_vec_size, const size_t block_size,
                                cudaStream_t stream) {
  const size_t grid_size = (len * emb_vec_size + block_size - 1) / block_size;
  fill_default_emb_vec_kernel<<<grid_size, block_size, 0, stream>>>(d_output, default_value, d_index, len, emb_vec_size);
}

void decompress_emb_vec_async(const float* d_input, const uint64_t* d_index, float* d_output,
                              const size_t len, const size_t emb_vec_size, const size_t block_size,
                              cudaStream_t stream) {
  const size_t grid_size = (len * emb_vec_size + block_size - 1) / block_size;
  decompress_emb_vec_kernel<<<grid_size, block_size, 0, stream>>>(d_input, d_index, d_output, len, emb_vec_size);
}

void generate_test_data(/* std::vector<float>& */ float* data, size_t size) {
  /* std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dis(-1.0f, 1.0f); */
  // 种子已在调用点设置
  for (size_t i = 0; i < size; i++) {
    data[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
  }
}

void generate_index_data(/* std::vector<uint64_t>& */ uint64_t* data, size_t size, size_t max_value) {
  /* std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<uint64_t> dis(0, max_value - 1); */
  // 注意：不在这里设置srand，因为generate_test_data已经设置了种子
  for (size_t i = 0; i < size; i++) {
    data[i] = rand() % max_value;
  }
}

bool compare_vectors(/* const std::vector<float>& */ const float* vec1, /* const std::vector<float>& */ const float* vec2, size_t size, float epsilon = 1e-5) {
  for (size_t i = 0; i < size; i++) {
    if (/* std::abs */ fabsf(vec1[i] - vec2[i]) > epsilon) return false;
  }
  return true;
}

void performance_comparison_test() {
    const size_t len = TEST_LEN;
    const size_t emb_vec_size = TEST_EMB_VEC_SIZE;

    // 输出测试配置
    printf("=== CPU算子性能比较测试 ===\n");
    printf("测试配置: len=%zu, emb_vec_size=%zu\n", len, emb_vec_size);
    printf("注意: 所有时间测量基于相同的数据量以确保可比性\n");

    // 分配内存
    float* h_input = (float*)malloc(len * emb_vec_size * sizeof(float));
    float* h_output_original = (float*)malloc(len * emb_vec_size * sizeof(float));
    float* h_output_optimized = (float*)malloc(len * emb_vec_size * sizeof(float));
    float* h_output_advanced = (float*)malloc(len * emb_vec_size * sizeof(float));
    uint64_t* h_index = (uint64_t*)malloc(len * sizeof(uint64_t));

    // 确保每次测试使用相同的数据
    srand(RANDOM_SEED);
    generate_test_data(h_input, len * emb_vec_size);
    generate_index_data(h_index, len, len);

    // 测试merge_emb_vec函数
    printf("\n--- merge_emb_vec 性能比较 ---\n");

    // 原始版本
    clock_t start = clock();
    merge_emb_vec_cpu(h_output_original, h_input, h_index, len, emb_vec_size);
    clock_t end = clock();
    double original_time = ((double)(end - start)) / CLOCKS_PER_SEC * 1000000;

    // 优化版本
    start = clock();
    merge_emb_vec_cpu_optimized(h_output_optimized, h_input, h_index, len, emb_vec_size);
    end = clock();
    double optimized_time = ((double)(end - start)) / CLOCKS_PER_SEC * 1000000;

    // 高级版本
    start = clock();
    merge_emb_vec_cpu_advanced(h_output_advanced, h_input, h_index, len, emb_vec_size);
    end = clock();
    double advanced_time = ((double)(end - start)) / CLOCKS_PER_SEC * 1000000;

    // 验证结果一致性
    bool optimized_match = compare_vectors(h_output_original, h_output_optimized, len * emb_vec_size);
    bool advanced_match = compare_vectors(h_output_original, h_output_advanced, len * emb_vec_size);

    printf("原始版本时间: %.2f 微秒\n", original_time);
    printf("优化版本时间: %.2f 微秒 (加速比: %.2fx, 结果匹配: %s)\n",
           optimized_time, original_time / optimized_time, optimized_match ? "是" : "否");
    printf("高级版本时间: %.2f 微秒 (加速比: %.2fx, 结果匹配: %s)\n",
           advanced_time, original_time / advanced_time, advanced_match ? "是" : "否");

    // 测试fill_default_emb_vec函数
    printf("\n--- fill_default_emb_vec 性能比较 ---\n");
    float default_value = 0.5f;

    // 原始版本
    start = clock();
    fill_default_emb_vec_cpu(h_output_original, default_value, h_index, len, emb_vec_size);
    end = clock();
    original_time = ((double)(end - start)) / CLOCKS_PER_SEC * 1000000;

    // 优化版本
    start = clock();
    fill_default_emb_vec_cpu_optimized(h_output_optimized, default_value, h_index, len, emb_vec_size);
    end = clock();
    optimized_time = ((double)(end - start)) / CLOCKS_PER_SEC * 1000000;

    // 高级版本
    start = clock();
    fill_default_emb_vec_cpu_advanced(h_output_advanced, default_value, h_index, len, emb_vec_size);
    end = clock();
    advanced_time = ((double)(end - start)) / CLOCKS_PER_SEC * 1000000;

    // 验证结果一致性
    optimized_match = compare_vectors(h_output_original, h_output_optimized, len * emb_vec_size);
    advanced_match = compare_vectors(h_output_original, h_output_advanced, len * emb_vec_size);

    printf("原始版本时间: %.2f 微秒\n", original_time);
    printf("优化版本时间: %.2f 微秒 (加速比: %.2fx, 结果匹配: %s)\n",
           optimized_time, original_time / optimized_time, optimized_match ? "是" : "否");
    printf("高级版本时间: %.2f 微秒 (加速比: %.2fx, 结果匹配: %s)\n",
           advanced_time, original_time / advanced_time, advanced_match ? "是" : "否");

    // 测试decompress_emb_vec函数
    printf("\n--- decompress_emb_vec 性能比较 ---\n");

    // 原始版本
    start = clock();
    decompress_emb_vec_cpu(h_input, h_index, h_output_original, len, emb_vec_size);
    end = clock();
    original_time = ((double)(end - start)) / CLOCKS_PER_SEC * 1000000;

    // 优化版本
    start = clock();
    decompress_emb_vec_cpu_optimized(h_input, h_index, h_output_optimized, len, emb_vec_size);
    end = clock();
    optimized_time = ((double)(end - start)) / CLOCKS_PER_SEC * 1000000;

    // 高级版本
    start = clock();
    decompress_emb_vec_cpu_advanced(h_input, h_index, h_output_advanced, len, emb_vec_size);
    end = clock();
    advanced_time = ((double)(end - start)) / CLOCKS_PER_SEC * 1000000;

    // 验证结果一致性
    optimized_match = compare_vectors(h_output_original, h_output_optimized, len * emb_vec_size);
    advanced_match = compare_vectors(h_output_original, h_output_advanced, len * emb_vec_size);

    printf("原始版本时间: %.2f 微秒\n", original_time);
    printf("优化版本时间: %.2f 微秒 (加速比: %.2fx, 结果匹配: %s)\n",
           optimized_time, original_time / optimized_time, optimized_match ? "是" : "否");
    printf("高级版本时间: %.2f 微秒 (加速比: %.2fx, 结果匹配: %s)\n",
           advanced_time, original_time / advanced_time, advanced_match ? "是" : "否");

    // 释放内存
    free(h_input);
    free(h_output_original);
    free(h_output_optimized);
    free(h_output_advanced);
    free(h_index);
}

void test_functions() {
  const size_t len = TEST_LEN;
  const size_t emb_vec_size = TEST_EMB_VEC_SIZE;
  const size_t BLOCK_SIZE = TEST_BLOCK_SIZE;

  // 输出GPU vs CPU测试配置
  printf("\n=== GPU vs CPU 性能对比测试 ===\n");
  printf("测试配置: len=%zu, emb_vec_size=%zu, block_size=%zu\n", len, emb_vec_size, BLOCK_SIZE);
  printf("使用高级优化版本的CPU函数进行对比\n");

  /* std::vector<float> h_input(len * emb_vec_size);
  std::vector<float> h_output_cpu(len * emb_vec_size);
  std::vector<float> h_output_gpu(len * emb_vec_size);
  std::vector<uint64_t> h_index(len); */
  float* h_input = (float*)malloc(len * emb_vec_size * sizeof(float));
  float* h_output_cpu = (float*)malloc(len * emb_vec_size * sizeof(float));
  float* h_output_gpu = (float*)malloc(len * emb_vec_size * sizeof(float));
  uint64_t* h_index = (uint64_t*)malloc(len * sizeof(uint64_t));

  // 确保每次测试使用相同的数据
  srand(RANDOM_SEED);
  generate_test_data(h_input, len * emb_vec_size);
  generate_index_data(h_index, len, len);

  float *d_input, *d_output;
  uint64_t *d_index;
  CHECK_CUDA(cudaMalloc(&d_input, len * emb_vec_size * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_output, len * emb_vec_size * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_index, len * sizeof(uint64_t)));

  cudaStream_t stream;
  CHECK_CUDA(cudaStreamCreate(&stream));

  MergeFunc merge_func = merge_emb_vec_async;
  FillDefaultFunc fill_default_func = fill_default_emb_vec_async;
  DecompressFunc decompress_func = decompress_emb_vec_async;

  /* std::cout << "\n测试 merge_emb_vec 函数:\n"; */
  printf("\n测试 merge_emb_vec 函数:\n");
  {
    /* auto start = std::chrono::high_resolution_clock::now(); */
    clock_t start = clock();
    merge_emb_vec_cpu_advanced(h_output_cpu, h_input, h_index, len, emb_vec_size);
    clock_t end = clock();
    /* auto end = std::chrono::high_resolution_clock::now();
    auto cpu_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count(); */
    double cpu_time = ((double)(end - start)) / CLOCKS_PER_SEC * 1000000;

    /* start = std::chrono::high_resolution_clock::now(); */
    CHECK_CUDA(cudaMemcpy(d_input, h_input, len * emb_vec_size * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_index, h_index, len * sizeof(uint64_t), cudaMemcpyHostToDevice));
    start = clock();

    merge_func(d_output, d_input, d_index, len, emb_vec_size, BLOCK_SIZE, stream);
    CHECK_CUDA(cudaStreamSynchronize(stream));
    end = clock();


    CHECK_CUDA(cudaMemcpy(h_output_gpu, d_output, len * emb_vec_size * sizeof(float), cudaMemcpyDeviceToHost));
    /* end = std::chrono::high_resolution_clock::now();
    auto gpu_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count(); */
    double gpu_time = ((double)(end - start)) / CLOCKS_PER_SEC * 1000000;

    bool results_match = compare_vectors(h_output_cpu, h_output_gpu, len * emb_vec_size);
    /* std::cout << "结果匹配: " << (results_match ? "是" : "否") << std::endl;
    std::cout << "CPU时间: " << cpu_time << " 微秒" << std::endl;
    std::cout << "GPU时间: " << gpu_time << " 微秒" << std::endl;
    std::cout << "加速比: " << static_cast<float>(cpu_time) / gpu_time << "x" << std::endl; */
    printf("结果匹配: %s\n", results_match ? "是" : "否");
    printf("CPU时间: %.2f 微秒\n", cpu_time);
    printf("GPU时间: %.2f 微秒\n", gpu_time);
    printf("加速比: %.2fx\n", (float)cpu_time / gpu_time);
  }

  /* std::cout << "\n测试 fill_default_emb_vec 函数:\n"; */
  printf("\n测试 fill_default_emb_vec 函数:\n");
  {
    float default_value = 0.5f;

    /* auto start = std::chrono::high_resolution_clock::now(); */
    clock_t start = clock();
    fill_default_emb_vec_cpu_advanced(h_output_cpu, default_value, h_index, len, emb_vec_size);
    clock_t end = clock();
    /* auto end = std::chrono::high_resolution_clock::now();
    auto cpu_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count(); */
    double cpu_time = ((double)(end - start)) / CLOCKS_PER_SEC * 1000000;

    /* start = std::chrono::high_resolution_clock::now(); */
    CHECK_CUDA(cudaMemcpy(d_index, h_index, len * sizeof(uint64_t), cudaMemcpyHostToDevice));
    start = clock();
    fill_default_func(d_output, default_value, d_index, len, emb_vec_size, BLOCK_SIZE, stream);
    end = clock();

    CHECK_CUDA(cudaStreamSynchronize(stream));
    CHECK_CUDA(cudaMemcpy(h_output_gpu, d_output, len * emb_vec_size * sizeof(float), cudaMemcpyDeviceToHost));
    /* end = std::chrono::high_resolution_clock::now();
    auto gpu_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count(); */
    double gpu_time = ((double)(end - start)) / CLOCKS_PER_SEC * 1000000;

    bool results_match = compare_vectors(h_output_cpu, h_output_gpu, len * emb_vec_size);
    /* std::cout << "结果匹配: " << (results_match ? "是" : "否") << std::endl;
    std::cout << "CPU时间: " << cpu_time << " 微秒" << std::endl;
    std::cout << "GPU时间: " << gpu_time << " 微秒" << std::endl;
    std::cout << "加速比: " << static_cast<float>(cpu_time) / gpu_time << "x" << std::endl; */
    printf("结果匹配: %s\n", results_match ? "是" : "否");
    printf("CPU时间: %.2f 微秒\n", cpu_time);
    printf("GPU时间: %.2f 微秒\n", gpu_time);
    printf("加速比: %.2fx\n", (float)cpu_time / gpu_time);
  }

  /* std::cout << "\n测试 decompress_emb_vec 函数:\n"; */
  printf("\n测试 decompress_emb_vec 函数:\n");
  {
    /* auto start = std::chrono::high_resolution_clock::now(); */
    clock_t start = clock();
    decompress_emb_vec_cpu_advanced(h_input, h_index, h_output_cpu, len, emb_vec_size);
    clock_t end = clock();
    /* auto end = std::chrono::high_resolution_clock::now();
    auto cpu_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count(); */
    double cpu_time = ((double)(end - start)) / CLOCKS_PER_SEC * 1000000;

    /* start = std::chrono::high_resolution_clock::now(); */
    CHECK_CUDA(cudaMemcpy(d_input, h_input, len * emb_vec_size * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_index, h_index, len * sizeof(uint64_t), cudaMemcpyHostToDevice));
    start = clock();
    decompress_func(d_input, d_index, d_output, len, emb_vec_size, BLOCK_SIZE, stream);
    end = clock();

    CHECK_CUDA(cudaStreamSynchronize(stream));
    CHECK_CUDA(cudaMemcpy(h_output_gpu, d_output, len * emb_vec_size * sizeof(float), cudaMemcpyDeviceToHost));
    /* end = std::chrono::high_resolution_clock::now();
    auto gpu_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count(); */
    double gpu_time = ((double)(end - start)) / CLOCKS_PER_SEC * 1000000;

    bool results_match = compare_vectors(h_output_cpu, h_output_gpu, len * emb_vec_size);
    /* std::cout << "结果匹配: " << (results_match ? "是" : "否") << std::endl;
    std::cout << "CPU时间: " << cpu_time << " 微秒" << std::endl;
    std::cout << "GPU时间: " << gpu_time << " 微秒" << std::endl;
    std::cout << "加速比: " << static_cast<float>(cpu_time) / gpu_time << "x" << std::endl; */
    printf("结果匹配: %s\n", results_match ? "是" : "否");
    printf("CPU时间: %.2f 微秒\n", cpu_time);
    printf("GPU时间: %.2f 微秒\n", gpu_time);
    printf("加速比: %.2fx\n", (float)cpu_time / gpu_time);
  }

  CHECK_CUDA(cudaFree(d_input));
  CHECK_CUDA(cudaFree(d_output));
  CHECK_CUDA(cudaFree(d_index));
  CHECK_CUDA(cudaStreamDestroy(stream));

  free(h_input);
  free(h_output_cpu);
  free(h_output_gpu);
  free(h_index);
}

int main() {
  performance_comparison_test();
  test_functions();
  return 0;
}


