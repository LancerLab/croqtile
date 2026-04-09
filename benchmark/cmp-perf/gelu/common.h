#include "cmath"

inline float gelu_cpu(float x) {
  return 0.5 * x * (1.0 + std::erf(x / std::sqrt(2.0)));
}

void test3(std::string name, std::function<spanned_data<float, 3>(spanned_view<float, 3UL>)> f, std::vector<int> dims) {
  auto input = choreo::make_spandata<choreo::f32>(dims[0], dims[1], dims[2]);
  input.fill_random(-1.0f, 1.0f);

  auto start = std::chrono::high_resolution_clock::now();
  auto res = f(input.view());
  auto end = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

  for (int i = 0; i < res.shape()[0]; ++i)
    for (int j = 0; j < res.shape()[1]; ++j)
      for (int k = 0; k < res.shape()[2]; ++k) {
        auto in = input[i][j][k];
        auto expect = gelu_cpu(in);
        auto actual = res[i][j][k];
        if (std::fabs(actual - expect) > 1e-3) {
          std::cerr << "[" << i << "," << j << "," << k << "]:\n";
          std::cerr << "\t" << "input : " << in << "\n";
          std::cerr << "\t" << "expect: " << expect << "\n";
          std::cerr << "\t" << "actual: " << actual << "\n";
          assert(false);
        }
      }
  std::cout << name << " is PASS.\n";
  std::cout << "Execution time: " << duration.count() << " microseconds" << std::endl;
}

void test4(std::string name, std::function<spanned_data<float, 4>(spanned_view<float, 4UL>)> f, std::vector<int> dims) {
  auto input = choreo::make_spandata<choreo::f32>(dims[0], dims[1], dims[2], dims[3]);
  input.fill_random(-1.0f, 1.0f);

  auto start = std::chrono::high_resolution_clock::now();
  auto res = f(input.view());
  auto end = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

  for (int i = 0; i < res.shape()[0]; ++i)
    for (int j = 0; j < res.shape()[1]; ++j)
      for (int k = 0; k < res.shape()[2]; ++k)
        for (int m = 0; m < res.shape()[3]; ++m) {
          auto in = input[i][j][k][m];
          auto expect = gelu_cpu(in);
          auto actual = res[i][j][k][m];
          if (std::fabs(actual - expect) > 1e-3) {
            std::cerr << "[" << i << "," << j << "," << k << "," << m << "]:\n";
            std::cerr << "\t" << "input : " << in << "\n";
            std::cerr << "\t" << "expect: " << expect << "\n";
            std::cerr << "\t" << "actual: " << actual << "\n";
            assert(false);
          }
        }
  std::cout << name << " is PASS.\n";
  std::cout << "Execution time: " << duration.count() << " microseconds" << std::endl;
}
