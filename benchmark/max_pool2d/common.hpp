#include "cmath"

#ifdef __CHECK__
#define check true
#else
#define check false
#endif

extern "C" inline void cpu_max_pool2d1(float* input, float* output, int N,
                                       int C, int H, int W) {
  const int out_H = H / 2;
  const int out_W = W / 2;

  for (int n = 0; n < N; ++n) {
    for (int c = 0; c < C; ++c) {
      for (int oh = 0; oh < out_H; ++oh) {
        for (int ow = 0; ow < out_W; ++ow) {
          int ih = oh * 2;
          int iw = ow * 2;

          int input_base = n * C * H * W + c * H * W;
          int output_idx =
              n * C * out_H * out_W + c * out_H * out_W + oh * out_W + ow;

          float max_val = input[input_base + ih * W + iw];
          max_val = std::max(max_val, input[input_base + (ih + 1) * W + iw]);
          max_val = std::max(max_val, input[input_base + ih * W + (iw + 1)]);
          max_val =
              std::max(max_val, input[input_base + (ih + 1) * W + (iw + 1)]);
          output[output_idx] = max_val;
        }
      }
    }
  }
}
extern "C" inline void cpu_max_pool2d2(float* input, float* output, int N,
                                       int C, int H, int W, int KH, int KW) {
  const int out_H = H / KH;
  const int out_W = W / KW;
  for (int n = 0; n < N; ++n) {
    for (int c = 0; c < C; ++c) {
      for (int oh = 0; oh < out_H; ++oh) {
        for (int ow = 0; ow < out_W; ++ow) {
          int ih = oh * KH;
          int iw = ow * KW;
          int input_base = n * C * H * W + c * H * W;
          int output_idx =
              n * C * out_H * out_W + c * out_H * out_W + oh * out_W + ow;
          float max_val = input[input_base + ih * W + iw];
          for (int kh = 0; kh < KH; ++kh) {
            for (int kw = 0; kw < KW; ++kw) {
              max_val = std::max(max_val,
                                 input[input_base + (ih + kh) * W + (iw + kw)]);
            }
          }
          output[output_idx] = max_val;
        }
      }
    }
  }
}

#define TESTP(func, N, C, H, W)                                                \
  auto input_##func = choreo::make_spandata<float>(N, C, H, W);                \
  input_##func.fill_random(-10.0f, 10.0f);                                     \
  auto start_##func = std::chrono::high_resolution_clock::now();               \
  auto output_##func = func(input_##func.view());                              \
  auto end_##func = std::chrono::high_resolution_clock::now();                 \
  if (check) {                                                                 \
    float* cpu_res_##func =                                                    \
        (float*)malloc(N * C * (H / 2) * (W / 2) * sizeof(float));             \
    cpu_max_pool2d1(input_##func.data(), cpu_res_##func, N, C, H, W);          \
    for (int i = 0; i < N; i++) {                                              \
      for (int j = 0; j < C; j++) {                                            \
        for (int h = 0; h < H / 2; h++) {                                      \
          for (int w = 0; w < W / 2; w++) {                                    \
            int idx = i * C * (H / 2) * (W / 2) + j * (H / 2) * (W / 2) +      \
                      h * (W / 2) + w;                                         \
            choreo::choreo_assert(                                             \
                fabs(output_##func[i][j][h][w] - cpu_res_##func[idx]) < 1e-3,  \
                "error");                                                      \
          }                                                                    \
        }                                                                      \
      }                                                                        \
    }                                                                          \
    printf("Test %s passed!\n", #func);                                        \
  }                                                                            \
  auto duration_##func =                                                       \
      std::chrono::duration_cast<std::chrono::microseconds>(end_##func -       \
                                                            start_##func);     \
  std::cout << "Case " << #func                                                \
            << " Execution time: " << duration_##func.count()                  \
            << " microseconds" << std::endl;

#define TESTP2(func, N, C, H, W, KH, KW)                                       \
  auto input_##func = choreo::make_spandata<float>(N, C, H, W);                \
  input_##func.fill_random(-10.0f, 10.0f);                                     \
  auto start_##func = std::chrono::high_resolution_clock::now();               \
  auto output_##func = func(input_##func.view());                              \
  auto end_##func = std::chrono::high_resolution_clock::now();                 \
  if (check) {                                                                 \
    float* cpu_res_##func =                                                    \
        (float*)malloc(N * C * (H / KH) * (W / KW) * sizeof(float));           \
    cpu_max_pool2d2(input_##func.data(), cpu_res_##func, N, C, H, W, KH, KW);  \
    for (int i = 0; i < N; i++) {                                              \
      for (int j = 0; j < C; j++) {                                            \
        for (int h = 0; h < H / KH; h++) {                                     \
          for (int w = 0; w < W / KW; w++) {                                   \
            int idx = i * C * (H / KH) * (W / KW) + j * (H / KH) * (W / KW) +  \
                      h * (W / KW) + w;                                        \
            choreo::choreo_assert(                                             \
                fabs(output_##func[i][j][h][w] - cpu_res_##func[idx]) < 1e-3,  \
                "error");                                                      \
          }                                                                    \
        }                                                                      \
      }                                                                        \
    }                                                                          \
    printf("Test %s passed!\n", #func);                                        \
  }                                                                            \
  auto duration_##func =                                                       \
      std::chrono::duration_cast<std::chrono::microseconds>(end_##func -       \
                                                            start_##func);     \
  std::cout << "Case " << #func                                                \
            << " Execution time: " << duration_##func.count()                  \
            << " microseconds" << std::endl;