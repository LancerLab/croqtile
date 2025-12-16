__cok__ {
  __co_device__ inline float v_dot(float* lhs, float* rhs, int elem_count) {
    // Use TCLE leaptr for vectorized memory access (float4 = 4*4=16 bytes)
    auto lhs_ptr = tcle::leaptr<__vector float, 1>(lhs, sizeof(__vector float));
    auto rhs_ptr = tcle::leaptr<__vector float, 1>(rhs, sizeof(__vector float));

    int vector_length = sizeof(__vector float) / sizeof(float);
    int vector_count = elem_count / vector_length;

    __vector float vresult = (__vector float)0;
#pragma unroll
    for (int i = 0; i < vector_count; ++i) {
      __vector float lhs_vec = lhs_ptr.load<0>();
      __vector float rhs_vec = rhs_ptr.load<0>();
      vresult += lhs_vec * rhs_vec;
    }

    float result = 0;
    if (vector_count > 0)
      for (int i = 0; i < vector_length; ++i) result += vresult[i];

    int offset = vector_count * vector_length;
    for (int i = 0; i < elem_count % vector_length; ++i)
      result += lhs[offset + i] * rhs[offset + i];

    return result;
  }

  __co_device__ extern "C" void k_matmul(
      __private__ float* lhs, __private__ float* rhs, __private__ float* out,
      int m, int n, int k) {
    for (int i = 0; i < m; ++i)
      for (int j = 0; j < n; ++j) {
        float res = 0;
#ifndef NO_VRED
        res = v_dot(&lhs[i * k], &rhs[j * k], k);
#else
        for (int z = 0; z < k; ++z) res += lhs[i * k + z] * rhs[j * k + z];
#endif
        out[i * n + j] += res;
      }
  }
}

extern "C" inline void cpu_matmul1(float* lhs_data, float* rhs_data,
                                   float* output_data, int H, int W, int K) {
  for (int i = 0; i < H; ++i) {
    for (int j = 0; j < W; ++j) {
      output_data[i * W + j] = 0.0f;
      for (int k = 0; k < K; ++k) {
        output_data[i * W + j] += lhs_data[i * K + k] * rhs_data[k * W + j];
      }
    }
  }
}

extern "C" inline void cpu_matmul2(float* lhs_data, float* rhs_data,
                                   float* output_data, int C, int H, int W,
                                   int K) {
  for (int c = 0; c < C; ++c) {
    for (int h = 0; h < H; ++h) {
      for (int w = 0; w < W; ++w) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
          int lhs_idx = c * H * K + h * K + k;
          int rhs_idx = k * W + w;
          sum += lhs_data[lhs_idx] * rhs_data[rhs_idx];
        }
        int out_idx = c * H * W + h * W + w;
        output_data[out_idx] = sum;
      }
    }
  }
}

extern "C" inline void cpu_matmul3(float* lhs_data, float* rhs_data,
                                   float* output_data, int N, int C, int H,
                                   int W, int K) {
  for (int n = 0; n < N; ++n) {
    for (int c = 0; c < C; ++c) {
      for (int h = 0; h < H; ++h) {
        for (int w = 0; w < W; ++w) {
          float sum = 0.0f;
          for (int k = 0; k < K; ++k) {
            int lhs_idx = n * C * H * K + c * H * K + h * K + k;
            int rhs_idx = n * C * K * W + c * K * W + k * W + w;
            sum += lhs_data[lhs_idx] * rhs_data[rhs_idx];
          }
          int out_idx = n * C * H * W + c * H * W + h * W + w;
          output_data[out_idx] = sum;
        }
      }
    }
  }
}

#ifdef __CHECK__
#define check true
#else
#define check false
#endif

#define TEST1(func, H, W, K)                                                   \
  auto lhs_##func = choreo::make_spandata<float>(H, K);                        \
  auto rhs_##func = choreo::make_spandata<float>(K, W);                        \
  lhs_##func.fill_random(-10.0f, 10.0f);                                       \
  rhs_##func.fill_random(-10.0f, 10.0f);                                       \
  auto start_##func = std::chrono::high_resolution_clock::now();               \
  auto res_##func = func(lhs_##func.view(), rhs_##func.view());                \
  auto end_##func = std::chrono::high_resolution_clock::now();                 \
  if (check) {                                                                 \
    float* cpu_res_##func = (float*)malloc(H * W * sizeof(float));             \
    cpu_matmul1(lhs_##func.data(), rhs_##func.data(), cpu_res_##func, H, W,    \
                K);                                                            \
    for (int i = 0; i < H; ++i) {                                              \
      for (int j = 0; j < W; ++j) {                                            \
        choreo::choreo_assert(                                                 \
            fabs(res_##func[i][j] - cpu_res_##func[i * W + j]) < 1e-1,         \
            "error");                                                          \
      }                                                                        \
    }                                                                          \
    printf("Case %s Passed!\n", #func);                                        \
  }                                                                            \
  auto duration_##func =                                                       \
      std::chrono::duration_cast<std::chrono::microseconds>(end_##func -       \
                                                            start_##func);     \
  std::cout << "Case " << #func                                                \
            << " Execution time: " << duration_##func.count()                  \
            << " microseconds" << std::endl;

#define TEST2(func, C, H, W, K)                                                \
  auto lhs_##func = choreo::make_spandata<float>(C, H, K);                     \
  auto rhs_##func = choreo::make_spandata<float>(K, W);                        \
  lhs_##func.fill_random(-10.0f, 10.0f);                                       \
  rhs_##func.fill_random(-10.0f, 10.0f);                                       \
  auto start_##func = std::chrono::high_resolution_clock::now();               \
  auto res_##func = func(lhs_##func.view(), rhs_##func.view());                \
  auto end_##func = std::chrono::high_resolution_clock::now();                 \
  if (check) {                                                                 \
    float* cpu_res_##func = (float*)malloc(C * H * W * sizeof(float));         \
    cpu_matmul2(lhs_##func.data(), rhs_##func.data(), cpu_res_##func, C, H, W, \
                K);                                                            \
    for (int c = 0; c < C; ++c) {                                              \
      for (int h = 0; h < H; ++h) {                                            \
        for (int w = 0; w < W; ++w) {                                          \
          choreo::choreo_assert(fabs(res_##func[c][h][w] -                     \
                                     cpu_res_##func[c * H * W + h * W + w]) <  \
                                    1e-1,                                      \
                                "error");                                      \
        }                                                                      \
      }                                                                        \
    }                                                                          \
    printf("Test %s Passed!\n", #func);                                        \
  }                                                                            \
  auto duration_##func =                                                       \
      std::chrono::duration_cast<std::chrono::microseconds>(end_##func -       \
                                                            start_##func);     \
  std::cout << "Case " << #func                                                \
            << " Execution time: " << duration_##func.count()                  \
            << " microseconds" << std::endl;

#define TEST3(func, N, C, H, W, K)                                             \
  auto lhs_##func = choreo::make_spandata<float>(N, C, H, K);                  \
  auto rhs_##func = choreo::make_spandata<float>(N, C, K, W);                  \
  lhs_##func.fill_random(-10.0f, 10.0f);                                       \
  rhs_##func.fill_random(-10.0f, 10.0f);                                       \
  auto start_##func = std::chrono::high_resolution_clock::now();               \
  auto res_##func = func(lhs_##func.view(), rhs_##func.view());                \
  auto end_##func = std::chrono::high_resolution_clock::now();                 \
  if (check) {                                                                 \
    float* cpu_res_##func = (float*)malloc(N * C * H * W * sizeof(float));     \
    cpu_matmul3(lhs_##func.data(), rhs_##func.data(), cpu_res_##func, N, C, H, \
                W, K);                                                         \
    for (int n = 0; n < N; ++n) {                                              \
      for (int c = 0; c < C; ++c) {                                            \
        for (int h = 0; h < H; ++h) {                                          \
          for (int w = 0; w < W; ++w) {                                        \
            choreo::choreo_assert(                                             \
                fabs(res_##func[n][c][h][w] -                                  \
                     cpu_res_##func[n * C * H * W + c * H * W + h * W + w]) <  \
                    1e-1,                                                      \
                "error");                                                      \
          }                                                                    \
        }                                                                      \
      }                                                                        \
    }                                                                          \
    printf("Test %s Passed!\n", #func);                                        \
  }                                                                            \
  auto duration_##func =                                                       \
      std::chrono::duration_cast<std::chrono::microseconds>(end_##func -       \
                                                            start_##func);     \
  std::cout << "Case " << #func                                                \
            << " Execution time: " << duration_##func.count()                  \
            << " microseconds" << std::endl;