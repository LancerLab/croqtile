#include <cassert>
#include <cstring>

float* embedding_cpu(
    const int* input,
    const float* weight,
    int batch_size, int token_size, int vocab_size, int hidden_size) {
  int out_size = batch_size * token_size * hidden_size;
  float* output = new float[out_size];

  for (int b = 0; b < batch_size; ++b) {
    for (int t = 0; t < token_size; ++t) {
      int idx = static_cast<int>(input[b * token_size + t]);
      if (idx < 0 || idx >= vocab_size)
        assert(false && "expect idx -> [0, vocab_size)");
      const float* w_row = weight + idx * hidden_size;
      float* out_ptr = output + (b * token_size + t) * hidden_size;
      std::memcpy(out_ptr, w_row, sizeof(float) * hidden_size);
    }
  }

  return output;
}

void test(std::string name, std::function<spanned_data<float, 3>(spanned_view<int, 2UL>, spanned_view<float, 2UL>)> f, std::vector<int> ids, std::vector<int> wds) {
  auto input = choreo::make_spandata<choreo::s32>(ids[0], ids[1]);
  input.fill_random(0, wds[0] - 1);

  auto weight = choreo::make_spandata<choreo::f32>(wds[0], wds[1]);
  weight.fill_random(-1.0f, 1.0f);

  auto start = std::chrono::high_resolution_clock::now();
  auto res = f(input.view(), weight.view());
  auto end = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
  
  auto res_cpu = embedding_cpu(input.data(), weight.data(), ids[0], ids[1], wds[0], wds[1]);

  auto nearlyEqual = [](float a, float b,
                 float absEps = 1e-3f, float relEps = 1e-2f) {
    float diff = std::fabs(a - b);
    if (diff <= absEps) return true;
    return diff <= relEps * std::max(std::fabs(a), std::fabs(b));
  };

  size_t idx = 0;
  for (int i = 0; i < res.shape()[0]; ++i)
    for (int j = 0; j < res.shape()[1]; ++j)
      for (int k = 0; k < res.shape()[2]; ++k, ++idx) {
        auto expect = res_cpu[idx];
        auto actual = res[i][j][k];
        if (!nearlyEqual(actual, expect)) {
          std::cerr << "[" << i << "," << j << "," << k << "]:\n";
          std::cerr << "\t" << "expect: " << expect << "\n";
          std::cerr << "\t" << "actual: " << actual << "\n";
          choreo::choreo_assert(false, "actual != expect");
        }
      }
  delete[] res_cpu;
  std::cout << name << " is PASS.\n";
  std::cout << "Execution time: " << duration.count() << " microseconds" << std::endl;
}