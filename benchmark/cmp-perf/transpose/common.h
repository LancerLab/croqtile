#include "cmath"

void test3(std::string name, std::function<spanned_data<float, 3>(spanned_view<float, 3UL>)> f, std::vector<int> dims, std::vector<size_t> td) {
  auto input = choreo::make_spandata<choreo::f32>(dims[0], dims[1], dims[2]);
  input.fill_random(-1.0f, 1.0f);

  auto start = std::chrono::high_resolution_clock::now();
  auto res = f(input.view());
  auto end = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

  for (size_t i = 0; i < input.shape()[0]; ++i)
    for (size_t j = 0; j < input.shape()[1]; ++j)
      for (size_t k = 0; k < input.shape()[2]; ++k) {
        size_t idx[] = {i, j, k};
        auto expect = input[i][j][k];
        auto actual = res[idx[td[0]]][idx[td[1]]][idx[td[2]]];
        if (std::fabs(actual - expect) > 1e-3) {
          std::cerr << "[" << i << "," << j << "," << k << "]:\n";
          std::cerr << "\t" << "expect: " << expect << "\n";
          std::cerr << "\t" << "actual: " << actual << "\n";
          choreo::choreo_assert(false, "error");
        }
      }
  std::cout << name << " is PASS.\n";
  std::cout << "Execution time: " << duration.count() << " microseconds" << std::endl;
}

void test4(std::string name, std::function<spanned_data<float, 4>(spanned_view<float, 4UL>)> f, std::vector<int> dims, std::vector<size_t> td) {
  auto input = choreo::make_spandata<choreo::f32>(dims[0], dims[1], dims[2], dims[3]);
  input.fill_random(-1.0f, 1.0f);

  auto start = std::chrono::high_resolution_clock::now();
  auto res = f(input.view());
  auto end = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

  for (size_t i = 0; i < input.shape()[0]; ++i)
    for (size_t j = 0; j < input.shape()[1]; ++j)
      for (size_t k = 0; k < input.shape()[2]; ++k)
        for (size_t m = 0; m < input.shape()[3]; ++m) {
          size_t idx[] = {i, j, k, m};
          auto expect = input[i][j][k][m];
          auto actual = res[idx[td[0]]][idx[td[1]]][idx[td[2]]][idx[td[3]]];
          if (std::fabs(actual - expect) > 1e-3) {
            std::cerr << "[" << i << "," << j << "," << k << "," << m << "]:\n";
            std::cerr << "\t" << "expect: " << expect << "\n";
            std::cerr << "\t" << "actual: " << actual << "\n";
            choreo::choreo_assert(false, "error");
          }
        }
  std::cout << name << " is PASS.\n";
  std::cout << "Execution time: " << duration.count() << " microseconds" << std::endl;
}