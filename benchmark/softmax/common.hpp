
#include <cmath>
extern "C" inline void cpu_softmax1(float* input, float* output, int H, int W,
                                    int C) {
  for (int i = 0; i < H; ++i) {
    for (int j = 0; j < W; ++j) {
      float max_val = input[i * W * C + j * C];
      for (int k = 1; k < C; ++k) {
        if (input[i * W * C + j * C + k] > max_val)
          max_val = input[i * W * C + j * C + k];
      }
      float sum_exp = 0.0f;
      for (int k = 0; k < C; ++k) {
        output[i * W * C + j * C + k] =
            expf(input[i * W * C + j * C + k] - max_val);
        sum_exp += output[i * W * C + j * C + k];
      }
      for (int k = 0; k < C; ++k) { output[i * W * C + j * C + k] /= sum_exp; }
    }
  }
}

extern "C" inline void cpu_softmax2(float* input, float* output, int N, int C,
                                    int H, int W) {
  const int total_elements = N * C * H * W;
  float max_val, sum_exp;
  for (int n = 0; n < N; ++n) {
    for (int h = 0; h < H; ++h) {
      for (int w = 0; w < W; ++w) {
        int base_idx = n * C * H * W + h * W + w;
        int stride = H * W;
        max_val = input[base_idx];
        for (int c = 1; c < C; ++c) {
          float val = input[base_idx + c * stride];
          if (val > max_val) max_val = val;
        }
        sum_exp = 0.0f;
        for (int c = 0; c < C; ++c) {
          float val = input[base_idx + c * stride];
          float exp_val = expf(val - max_val);
          output[base_idx + c * stride] = exp_val;
          sum_exp += exp_val;
        }
        for (int c = 0; c < C; ++c) {
          output[base_idx + c * stride] /= sum_exp;
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

#define TEST1(func, H, W, C)                                                   \
  auto input_##func = choreo::make_spandata<float>(H, W, C);                   \
  input_##func.fill_random(-10.0f, 10.0f);                                     \
  auto start_##func = std::chrono::high_resolution_clock::now();               \
  auto output_##func = func(input_##func.view());                              \
  auto end_##func = std::chrono::high_resolution_clock::now();                 \
  if (check) {                                                                 \
    float* cpu_res_##func = (float*)malloc(H * W * C * sizeof(float));         \
    cpu_softmax1(input_##func.data(), cpu_res_##func, H, W, C);                \
    for (int h = 0; h < H; ++h)                                                \
      for (int w = 0; w < W; ++w)                                              \
        for (int c = 0; c < C; ++c)                                            \
          if (fabs(output_##func[h][w][c] -                                    \
                   cpu_res_##func[h * W * C + w * C + c]) > 1e-3) {            \
            choreo::choreo_assert(false, "Test Failed");                       \
          }                                                                    \
    printf("Test %s passed!\n", #func);                                        \
  }                                                                            \
  auto duration_##func =                                                       \
      std::chrono::duration_cast<std::chrono::microseconds>(end_##func -       \
                                                            start_##func);     \
  std::cout << "Case " << #func                                                \
            << " Execution time: " << duration_##func.count()                  \
            << " microseconds" << std::endl;

#define TEST2(func, N, C, H, W)                                                \
  auto input_##func = choreo::make_spandata<float>(N, C, H, W);                \
  input_##func.fill_random(-1.0f, 1.0f);                                       \
  auto start_##func = std::chrono::high_resolution_clock::now();               \
  auto output_##func = func(input_##func.view());                              \
  auto end_##func = std::chrono::high_resolution_clock::now();                 \
  if (check) {                                                                 \
    float* cpu_res_##func = (float*)malloc(N * C * H * W * sizeof(float));     \
    cpu_softmax2(input_##func.data(), cpu_res_##func, N, C, H, W);             \
    for (int n = 0; n < N; ++n)                                                \
      for (int h = 0; h < H; ++h)                                              \
        for (int w = 0; w < W; ++w)                                            \
          for (int c = 0; c < C; ++c)                                          \
            if (fabs(output_##func[n][c][h][w] -                               \
                     cpu_res_##func[n * C * H * W + c * H * W + h * W + w]) >  \
                1e-3) {                                                        \
              choreo::choreo_assert(false, "Test Failed");                     \
            }                                                                  \
    printf("Test %s passed!\n", #func);                                        \
  }                                                                            \
  auto duration_##func =                                                       \
      std::chrono::duration_cast<std::chrono::microseconds>(end_##func -       \
                                                            start_##func);     \
  std::cout << "Case " << #func                                                \
            << " Execution time: " << duration_##func.count()                  \
            << " microseconds" << std::endl;