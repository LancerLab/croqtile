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

#define TEST1(func, H, W, K)                                                   \
  auto lhs_##func = choreo::make_spandata<float>(H, K);                        \
  auto rhs_##func = choreo::make_spandata<float>(K, W);                        \
  lhs_##func.fill_random(-10.0f, 10.0f);                                       \
  rhs_##func.fill_random(-10.0f, 10.0f);                                       \
  auto start_##func = std::chrono::high_resolution_clock::now();               \
  auto res_##func = func(lhs_##func.view(), rhs_##func.view());                \
  auto end_##func = std::chrono::high_resolution_clock::now();                 \
  float* cpu_res_##func = (float*)malloc(H * W * sizeof(float));               \
  cpu_matmul1(lhs_##func.data(), rhs_##func.data(), cpu_res_##func, H, W, K);  \
  for (int i = 0; i < H; ++i) {                                                \
    for (int j = 0; j < W; ++j) {                                              \
      assert(fabs(res_##func.at(i, j) - cpu_res_##func[i * W + j]) < 1e-3);    \
    }                                                                          \
  }                                                                            \
  printf("Case %s Passed!\n", #func);                                          \
  auto duration_##func =                                                       \
      std::chrono::duration_cast<std::chrono::microseconds>(end_##func -       \
                                                            start_##func);     \
  std::cout << "Execution time: " << duration_##func.count()                   \
            << " microseconds" << std::endl;

#define TEST2(func, C, H, W, K)                                                \
  auto lhs_##func = choreo::make_spandata<float>(C, H, K);                     \
  auto rhs_##func = choreo::make_spandata<float>(K, W);                        \
  lhs_##func.fill_random(-10.0f, 10.0f);                                       \
  rhs_##func.fill_random(-10.0f, 10.0f);                                       \
  auto start_##func = std::chrono::high_resolution_clock::now();               \
  auto res_##func = func(lhs_##func.view(), rhs_##func.view());                \
  float* cpu_res_##func = (float*)malloc(C * H * W * sizeof(float));           \
  auto end_##func = std::chrono::high_resolution_clock::now();                 \
  cpu_matmul2(lhs_##func.data(), rhs_##func.data(), cpu_res_##func, C, H, W,   \
              K);                                                              \
  for (int c = 0; c < C; ++c) {                                                \
    for (int h = 0; h < H; ++h) {                                              \
      for (int w = 0; w < W; ++w) {                                            \
        assert(fabs(res_##func.at(c, h, w) -                                   \
                    cpu_res_##func[c * H * W + h * W + w]) < 1e-3);            \
      }                                                                        \
    }                                                                          \
  }                                                                            \
  printf("Test %s Passed!\n", #func);                                          \
  auto duration_##func =                                                       \
      std::chrono::duration_cast<std::chrono::microseconds>(end_##func -       \
                                                            start_##func);     \
  std::cout << "Execution time: " << duration_##func.count()                   \
            << " microseconds" << std::endl;

#define TEST3(func, N, C, H, W, K)                                             \
  auto lhs_##func = choreo::make_spandata<float>(N, C, H, K);                  \
  auto rhs_##func = choreo::make_spandata<float>(N, C, K, W);                  \
  lhs_##func.fill_random(-10.0f, 10.0f);                                       \
  rhs_##func.fill_random(-10.0f, 10.0f);                                       \
  auto start_##func = std::chrono::high_resolution_clock::now();               \
  auto res_##func = func(lhs_##func.view(), rhs_##func.view());                \
  auto end_##func = std::chrono::high_resolution_clock::now();                 \
  float* cpu_res_##func = (float*)malloc(N * C * H * W * sizeof(float));       \
  cpu_matmul3(lhs_##func.data(), rhs_##func.data(), cpu_res_##func, N, C, H,   \
              W, K);                                                           \
  for (int n = 0; n < N; ++n) {                                                \
    for (int c = 0; c < C; ++c) {                                              \
      for (int h = 0; h < H; ++h) {                                            \
        for (int w = 0; w < W; ++w) {                                          \
          assert(fabs(res_##func.at(n, c, h, w) -                              \
                      cpu_res_##func[n * C * H * W + c * H * W + h * W + w]) < \
                 1e-3);                                                        \
        }                                                                      \
      }                                                                        \
    }                                                                          \
  }                                                                            \
  printf("Test %s Passed!\n", #func);                                          \
  auto duration_##func =                                                       \
      std::chrono::duration_cast<std::chrono::microseconds>(end_##func -       \
                                                            start_##func);     \
  std::cout << "Execution time: " << duration_##func.count()                   \
            << " microseconds" << std::endl;